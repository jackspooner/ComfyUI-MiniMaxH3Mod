from dataclasses import dataclass
import importlib
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import torch
import test_loaders as loaders
from comfy import execution_metrics
from app.run_timing_refmod_apply import observe_refmod_apply_metrics, refmod_apply_complete

N = loaders.N
CORE = importlib.import_module(N.__package__ + '.core')


class ApplicationTimingTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        previous = execution_metrics._observer
        execution_metrics.set_observer(lambda event, facts: self.events.append((event, facts)))
        self.addCleanup(execution_metrics.set_observer, previous)
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def mod(self, frames=1, audio=False):
        shape = (1, 32, 2, frames) if audio else (1, 24, frames, 4, 4)
        return N.H3RefMod(name='private reference', kind='audio' if audio else 'video' if frames > 1 else 'image',
                          latent=torch.arange(torch.tensor(shape).prod()).reshape(shape).float(),
                          latent_t=frames, latent_h=4, latent_w=4)

    def state(self):
        state = {'resolved': {}, 'measurements': {}, 'reasons': []}
        for event, facts in self.events:
            self.assertTrue(observe_refmod_apply_metrics(state, event, facts), event)
        self.assertTrue(refmod_apply_complete(state), state['reasons'])
        self.assertNotIn('private', json.dumps({key: state[key] for key in ('resolved', 'measurements', 'reasons')}))
        return state

    def test_actual_drop_reuse_scalar_and_frame_mix_preserve_outputs_and_call_counts(self):
        mods = [(self.mod(), 0), (self.mod(), 1), (self.mod(2), .5), (self.mod(3, audio=True), .5)]
        conditioning = [[torch.zeros(1, 1, 1), {'existing': True}]]
        with patch.object(CORE, '_blur_latent', wraps=CORE._blur_latent) as blur:
            result = N.MiniMaxH3RefModApply.execute(conditioning, mods, curve_value=.5)
            self.assertEqual(blur.call_count, 2)
        state = self.state()
        self.assertEqual([state['resolved'][f'refmodApply.reference.{i}.branch'] for i in range(4)],
                         ['drop', 'reuse', 'frame_mix', 'frame_mix'])
        self.assertEqual(len(result[0][0][1]['minimax_refs']), 3)
        self.assertIs(result[0][0][1]['minimax_refs'][0]['latent'], mods[1][0].latent)
        self.assertNotIn('minimax_refs', conditioning[0][1])
        baseline = [entry[0].latent.clone() for entry in mods]
        with patch.object(N.timing, '_metrics', None):
            quiet = N.MiniMaxH3RefModApply.execute(conditioning, mods, curve_value=.5)
        for got, expected in zip(result[0][0][1]['minimax_refs'], quiet[0][0][1]['minimax_refs']):
            key = 'audio_latent' if got['kind'] == 'audio' else 'latent'
            self.assertTrue(torch.equal(got[key], expected[key]))
        self.assertTrue(torch.equal(result[1], quiet[1]))
        for (mod, _), original in zip(mods, baseline):
            self.assertTrue(torch.equal(mod.latent, original))

    def test_scalar_single_image_curve_cap_scramble_and_saved_override(self):
        mods = [(self.mod(), 1), (self.mod(2), 1), (self.mod(4, audio=True), 1)]
        mods[0][0].config = {'curve': ['concept_at_end', 'linear', .3], 'retention': .8}
        N.MiniMaxH3RefModApply.execute([], mods, override=True, scramble_seed=7, scramble_mode='subset', scramble_keep=2)
        state = self.state()
        self.assertEqual(state['resolved']['refmodApply.inputs'], 3)
        self.assertEqual(state['resolved']['refmodApply.selected'], 2)
        self.assertTrue(state['resolved']['refmodApply.scrambled'])
        self.assertNotIn('scramble_seed', repr(state['resolved']))

    def test_real_debug_graph_and_temporary_preset_io_with_pack_conditioning(self):
        @dataclass
        class Conditioning:
            refs: list
        with tempfile.TemporaryDirectory() as directory, patch.object(N, '_graph_presets_dir', return_value=directory):
            N.MiniMaxH3RefModApply.execute(Conditioning([]), [], graph_preset='private missing', save_preset_as='private saved')
        state = self.state()
        self.assertEqual(state['resolved']['refmodApply.debug.samples'], 160)
        self.assertEqual(state['resolved']['refmodApply.conditioningEntries'], 1)
        self.assertTrue(state['measurements']['refmodApply.preset_read.succeeded'])
        self.assertTrue(state['measurements']['refmodApply.preset_save.succeeded'])

    def test_observer_failure_does_not_change_blur_or_repeat_original_errors(self):
        mod = self.mod()
        with patch.object(execution_metrics, 'emit', side_effect=RuntimeError('observer')):
            with patch.object(CORE, '_blur_latent', wraps=CORE._blur_latent) as blur:
                N.MiniMaxH3RefModApply.execute([], [(mod, .5)])
                self.assertEqual(blur.call_count, 1)
            with patch.object(CORE, '_blur_latent', side_effect=ValueError('original')) as blur:
                with self.assertRaisesRegex(ValueError, 'original'):
                    N.MiniMaxH3RefModApply.execute([], [(mod, .5)])
                self.assertEqual(blur.call_count, 1)

    def test_native_store_clears_only_application_gap_and_keeps_unknown_sources(self):
        from app.run_timing import RunTimingStore
        import app.run_timing as owner
        from pathlib import Path
        import jsonschema

        with tempfile.TemporaryDirectory() as directory:
            store = RunTimingStore(str(Path(directory) / 'timing.sqlite'))
            self.addCleanup(store.db.close)
            graph = {'loader': {'class_type': 'MiniMaxH3RefModsLoader', 'inputs': {}},
                     'conditioning': {'class_type': 'MiniMaxH3ReferenceToVideo', 'inputs': {'width': 32, 'height': 32, 'length': 6}},
                     'apply': {'class_type': 'MiniMaxH3RefModApply', 'inputs': {'conditioning': ['conditioning', 0], 'mods': ['loader', 0]}}}
            item = (1, 'prompt', graph, {}, ['apply'])
            store.event('accepted', item)
            store.event('started', item)
            store.safe_stage('prompt', 'apply', 'started')
            unexpected_writes = []
            def observe(event, facts):
                changes = store.db.total_changes
                store.safe_metric('prompt', 'apply', event, facts)
                if (event not in ('refmod_apply_begin', 'refmod_apply_end', 'refmod_apply_metadata_unavailable')
                        and store.db.total_changes != changes):
                    unexpected_writes.append(event)
            execution_metrics.set_observer(observe)
            N.MiniMaxH3RefModApply.execute([], [(self.mod(), .5)])
            self.assertEqual(unexpected_writes, [])
            store.safe_stage('prompt', 'apply', 'finished')
            page = store.page('0', 100, [], [])
            record = page['observations'][-1]
            stage = record['stages'][-1]
            self.assertTrue(stage['workload']['complete'], stage['reasons'])
            self.assertNotIn('refmod_apply_work_unmeasured', record['workload']['evidence']['reasons'])
            self.assertIn('refmod_loader_work_unmeasured', record['workload']['evidence']['reasons'])
            self.assertFalse(record['workload']['evidence']['complete'])
            schema = json.loads(Path(owner.__file__).with_name('run_timing.schema.json').read_text())
            jsonschema.validate(page, schema['$defs']['TimingObservationPage'])
            changes = store.db.total_changes
            store.page('0', 100, [], [])
            self.assertEqual(store.db.total_changes, changes)

    def test_telemetry_limit_preserves_every_original_reference_call(self):
        mod = self.mod()
        with patch.object(mod, 'ref_block', wraps=mod.ref_block) as build:
            result = N.MiniMaxH3RefModApply.execute([], [(mod, 0)] * 257)
            self.assertEqual(build.call_count, 257)
        state = {'resolved': {}, 'measurements': {}, 'reasons': []}
        for event, facts in self.events:
            observe_refmod_apply_metrics(state, event, facts)
        self.assertFalse(refmod_apply_complete(state))
        self.assertIn('refmod_apply_reference_limit', state['reasons'])
        self.assertEqual(len(state['refmod_apply']['references']), 256)
        self.assertEqual(result[0], [])
