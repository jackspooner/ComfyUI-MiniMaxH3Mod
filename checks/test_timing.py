"""Real temporary RefMods through the optional bridge and native timing owner."""
import json
import unittest
from unittest.mock import patch

import test_loaders as loaders
from comfy import execution_metrics
from app.run_timing import RunTimingStore
from app.run_timing_refmod import observe_refmod_loader_metrics, refmod_loader_complete

N = loaders.N


class TimingTests(unittest.TestCase):
    setUp = loaders.LoaderTests.setUp
    save_mod = loaders.LoaderTests.save_mod

    def record(self, function, **kwargs):
        events = []
        previous = execution_metrics._observer
        execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
        try:
            result = function(**kwargs)
        finally:
            execution_metrics.set_observer(previous)
        state = {'resolved': {}, 'measurements': {}, 'reasons': []}
        for event, facts in events:
            self.assertTrue(observe_refmod_loader_metrics(state, event, facts))
        self.assertTrue(refmod_loader_complete(state), state['reasons'])
        return result, events, state

    def test_actual_cache_branch_copies_replacement_and_private_metadata(self):
        path = self.save_mod('private/person', value=3)
        args = dict(mod_1='private/person', copies_1=3, strength_1=.4)
        with patch.object(N.H3RefMod, 'load', wraps=N.H3RefMod.load) as read:
            cold, events, state = self.record(N.MiniMaxH3RefModsLoader().load, **args)
            warm, _, warm_state = self.record(N.MiniMaxH3RefModsLoader().load, **args)
            self.assertEqual(read.call_count, 1)
            self.assertIs(cold[0][0][0], warm[0][0][0])
            self.assertIs(cold[0][0][0], cold[0][2][0])
            key = 'refmodLoader.slot.1.member.0.'
            self.assertFalse(state['resolved'][key + 'reused'])
            self.assertTrue(warm_state['resolved'][key + 'reused'])
            self.assertEqual(state['resolved']['refmodLoader.references'], 3)
            self.assertEqual(state['resolved'][key + 'tokens'], 4)
            self.assertEqual(state['measurements'][key + 'stamp_assurance'], 'size_mtime_ctime')
            self.save_mod('private/person', value=7)
            replaced, _, replaced_state = self.record(N.MiniMaxH3RefModsLoader().load, **args)
            self.assertEqual(read.call_count, 2)
            self.assertFalse(replaced_state['resolved'][key + 'reused'])
            self.assertEqual(replaced[0][0][0].latent.flatten()[0].item(), 7)
        serialized = repr(events) + json.dumps({key: state[key] for key in ('resolved', 'measurements', 'reasons')})
        self.assertNotIn(str(path), serialized)
        self.assertNotIn('private/person', serialized)
        self.assertNotIn('stamp_identity', repr(state['resolved']))
        self.assertNotIn('cpuSeconds', repr(state['resolved']))

    def test_skips_filtering_axis_and_no_extra_stamp_reads(self):
        self.save_mod('person')
        with patch.object(N, '_load_mod', wraps=N._load_mod) as read:
            _, _, state = self.record(N.MiniMaxH3RefModsLoader().load, mod_1='person', strength_1=0)
            self.assertEqual(read.call_count, 0)
            self.assertEqual(state['resolved']['refmodLoader.references'], 0)
            self.record(N.MiniMaxH3RefModsLoader().load, mod_1='person', components_1='Audio')
            self.assertEqual(read.call_count, 0)
        result, _, state = self.record(N.MiniMaxH3RefModsAxis().load, mod_a_2='person', value_2=-.25)
        self.assertEqual(result[0][0][1], .25)
        self.assertEqual(state['measurements']['refmodLoader.slot.2.side'], 'a')
        with patch.object(N, '_file_stamp', wraps=N._file_stamp) as stat:
            self.record(N.MiniMaxH3RefModsLoader().load, mod_1='person')
            self.assertEqual(stat.call_count, 1)

    def test_real_bundle_selection_uncached_and_ordered(self):
        import importlib
        import torch
        bundle = importlib.import_module(N.__package__ + '.bundle')
        visual = N.H3RefMod(name='secret visual', kind='image', latent=torch.ones(1, 24, 1, 4, 4))
        audio = N.H3RefMod(name='secret audio', kind='audio', latent=torch.ones(1, 32, 2, 3))
        bundle.save_bundle(str(self.root / 'combined'), 'secret bundle', [(visual, 1), (audio, 1)])
        args = dict(mod_1='combined', copies_1=2, audio_strength_1=.5)
        for _ in range(2):
            result, _, state = self.record(N.MiniMaxH3RefModsLoader().load, **args)
            self.assertEqual(len(result[0]), 4)
            self.assertEqual(state['measurements']['refmodLoader.slot.1.member.1.member'], 1)
            self.assertFalse(state['resolved']['refmodLoader.slot.1.member.0.reused'])
            self.assertEqual(state['resolved']['refmodLoader.slot.1.member.1.tokens'], 6)
        result, _, state = self.record(N.MiniMaxH3RefModsLoader().load, **args, components_1='Visual')
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(state['resolved']['refmodLoader.slot.1.members'], 1)
        self.assertNotIn('secret', repr(state))

    def test_fifo_and_sidecar_invalidation_stay_owned_by_existing_cache(self):
        self.save_mod('a')
        self.save_mod('b')
        with patch.object(N, '_MOD_CACHE_MAX', 1):
            self.record(N.MiniMaxH3RefModsLoader().load, mod_1='a', mod_2='b')
            _, _, state = self.record(N.MiniMaxH3RefModsLoader().load, mod_1='a')
            self.assertFalse(state['resolved']['refmodLoader.slot.1.member.0.reused'])
        (self.root / 'a.json').write_text('{"kind":"image"}')
        _, _, state = self.record(N.MiniMaxH3RefModsLoader().load, mod_1='a')
        self.assertFalse(state['resolved']['refmodLoader.slot.1.member.0.reused'])

    def test_absent_or_throwing_telemetry_preserves_result_and_single_failure(self):
        self.save_mod('a')
        with patch.object(N.timing, '_metrics', None):
            self.assertEqual(len(N.MiniMaxH3RefModsLoader().load(mod_1='a')[0]), 1)
        with patch.object(execution_metrics, '_observer', lambda *_: None), patch.object(execution_metrics, 'emit', side_effect=RuntimeError('telemetry')):
            self.assertEqual(len(N.MiniMaxH3RefModsLoader().load(mod_1='a')[0]), 1)
            N._MOD_CACHE.clear()
            with patch.object(N.H3RefMod, 'load', side_effect=ValueError('original failure')) as read:
                with self.assertRaisesRegex(ValueError, 'original failure'):
                    N.MiniMaxH3RefModsLoader().load(mod_1='a')
                self.assertEqual(read.call_count, 1)

    def test_real_native_store_retains_loader_inputs_without_inventing_a_model(self):
        self.save_mod('a')
        store = RunTimingStore(str(self.root / 'timing.sqlite'))
        self.addCleanup(store.db.close)
        graph = {'loader': {'class_type': 'MiniMaxH3RefModsLoader', 'inputs': {'mod_1': 'a'}}}
        item = (1, 'prompt', graph, {}, ['loader'])
        store.event('accepted', item)
        store.event('started', item)
        store.safe_stage('prompt', 'loader', 'started')
        previous = execution_metrics._observer
        execution_metrics.set_observer(lambda event, facts: store.safe_metric('prompt', 'loader', event, facts))
        try:
            N.MiniMaxH3RefModsLoader().load(mod_1='a')
        finally:
            execution_metrics.set_observer(previous)
        store.safe_stage('prompt', 'loader', 'finished')
        record = store.page('0', 100, [], [])['observations'][-1]
        stage = record['stages'][0]
        self.assertEqual(stage['kind'], 'loading')
        self.assertTrue(stage['workload']['complete'], stage['reasons'])
        self.assertNotIn('refmod_loader_work_unmeasured', record['workload']['evidence']['reasons'])
        self.assertIsNone(stage['workload']['modelIdentity'])
        self.assertEqual(stage['workload']['profile'], 'unknown')
        self.assertEqual(stage['workload']['resolved']['refmodLoader.references']['value'], 1)
        self.assertNotIn(str(self.root), json.dumps(record))
        import app.run_timing as owner
        import jsonschema
        from pathlib import Path
        schema = json.loads(Path(owner.__file__).with_name('run_timing.schema.json').read_text())
        jsonschema.validate(store.page('0', 100, [], []), schema['$defs']['TimingObservationPage'])
