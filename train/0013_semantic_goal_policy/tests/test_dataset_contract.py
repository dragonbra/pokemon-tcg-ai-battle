from __future__ import annotations
import copy,hashlib,importlib,json,tempfile,unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
records=importlib.import_module("train.0013_semantic_goal_policy.data.records");dataset=importlib.import_module("train.0013_semantic_goal_policy.data.dataset");source=importlib.import_module("train.0013_semantic_goal_policy.data.source");split=importlib.import_module("train.0013_semantic_goal_policy.data.split");protocol=importlib.import_module("train.0013_semantic_goal_policy.protocol");shards=importlib.import_module("train.0013_semantic_goal_policy.data.shards")
PROTOCOL=protocol.PreRunProtocol.from_json(Path(__file__).parents[1]/"configs"/"pre_run_protocol.json")
def observation(*,unknown=False):
    option={"type":1,"cardId":1,"serial":44};option.update({"mystery":7} if unknown else {})
    nested={"id":11,"serial":12,"energies":[101,102],"energyCards":[{"id":201,"serial":202}],"preEvolution":[{"id":301,"serial":302}],"tools":[{"id":401,"serial":402}]}
    return {"current":{"yourIndex":0,"firstPlayer":0,"turn":3,"turnActionCount":1,"energyAttached":False,"retreated":False,"stadiumPlayed":False,"supporterPlayed":False,"result":-1,"players":[{"active":[nested],"bench":[],"benchMax":5,"deckCount":50,"discard":[],"hand":[],"handCount":0,"prize":[]},{"active":[],"bench":[],"benchMax":5,"deckCount":50,"discard":[],"hand":[],"handCount":0,"prize":[]}],"stadium":[],"looking":[{"id":7,"serial":8}]},"select":{"type":0,"context":0,"contextCard":{"id":1,"serial":2},"deck":[{"id":3,"serial":4}],"effect":0,"remainDamageCounter":0,"remainEnergyCost":0,"minCount":1,"maxCount":1,"option":[option]},"logs":[{"type":1,"playerIndex":0,"serial":9}],"remainingOverageTime":60,"search_begin_input":None,"step":1}
def episode(*,episode_id=81,unknown=False,visual=True):
    deck0=[1]*30+[2]*30;deck1=[9]*60;raw_obs=observation(unknown=unknown);visual_obs=copy.deepcopy(raw_obs)
    visual_obs.pop("search_begin_input");visual_obs.pop("remainingOverageTime");visual_obs.pop("step")
    registration={"obs":{"current":{"yourIndex":0}},"selected":None,"action":[deck0,deck1]};decision={"obs":visual_obs,"selected":[0],"action":[[0],[]],"current":{"result":1},"select":{"type":99},"logs":[{"type":999}]};frames=[registration,decision]
    if visual:
        steps=[[{"status":"ACTIVE","observation":{"current":{"yourIndex":0}},"action":[],"visualize":frames},{"status":"ACTIVE","observation":{},"action":[]}],[{"status":"ACTIVE","observation":raw_obs,"action":[]},{"status":"ACTIVE","observation":{},"action":[]}],[{"status":"DONE","observation":{},"action":[0],"duration":2},{"status":"DONE","observation":{},"action":[]}]]
    else:
        steps=[[{"status":"ACTIVE","observation":None,"action":[]},{"status":"ACTIVE","observation":None,"action":[]}],[{"status":"ACTIVE","observation":raw_obs,"action":deck0},{"status":"ACTIVE","observation":{},"action":deck1}],[{"status":"DONE","observation":{},"action":[0],"duration":2},{"status":"DONE","observation":{},"action":[]}]]
    payload={"info":{"TeamNames":["Yushin Ito","Other"]},"steps":steps,"rewards":[1,-1],"statuses":["DONE","DONE"],"duration":10};identity=records.SourceIdentity("2026-07-24",episode_id,0);digest=hashlib.sha256((json.dumps(payload,sort_keys=True,separators=(",",":"))+"\n").encode()).hexdigest();trajectory=tuple(step[0] for step in steps);return source.CanonicalEpisode(identity,payload,trajectory,digest,"sanitized_real")
def manifest(e):
    registration=e.payload["steps"][0][0].get("visualize")
    cards=registration[0]["action"][0] if registration else e.payload["steps"][1][0]["action"]
    deck=dataset.DeckManifest.from_card_ids(cards);return split.assign_groups([split.SplitGroup("2026-07-24",deck.sha256,81,0),split.SplitGroup("2026-07-24",deck.sha256,82,0)],PROTOCOL)
def metadata(m):return {"source_manifest_sha256":dataset.SOURCE_MANIFEST_SHA256,"source_audit":{"seen":2,"eligible":2,"ineligible_noncomplete":0,"spool_bytes":1},"distribution_audit":{"group_counts_by_date":{"2026-07-24":2},"group_counts_by_deck_manifest_sha256":{},"decision_counts_by_date":{"2026-07-24":2},"decision_counts_by_deck_manifest_sha256":{},"decision_counts_by_action_type":{"1":2},"decision_counts_by_select_type":{"0":2},"cross_stratum_conflicts":{"comparable_state_groups":0,"conflicting_groups":0,"conflict_rate":0.0}},"split_private_assignments_sha256":m.private_sha256(),"split_audit_sha256":m.sha256(),"protocol_sha256":PROTOCOL.sha256(),"record_schema_version":dataset.SCHEMA_VERSION,"action_contract_version":dataset.ACTION_CONTRACT_VERSION}
class DatasetContractTest(unittest.TestCase):
    def test_sanitized_real_visual_registration_and_shifted_alignment(self):
        e=episode();record=list(dataset.iter_records(e,manifest(e),frozen_unknown_option_fields=frozenset()))[0];payload=record.as_dict();self.assertNotIn("value_target",payload);self.assertEqual(payload["ordered_action"],[0]);self.assertEqual(payload["deck_manifest"]["counts"],[[1,30],[2,30]]);self.assertNotIn("result",payload["actor_observation"]["current"]);self.assertIn("deck",payload["actor_observation"]["select"]);self.assertEqual(payload["actor_observation"]["logs"][0]["serial"],9);self.assertEqual(payload["event_cursor"],{"visual_frame_index":1,"actor_decision_index":0,"incoming_log_count":1});self.assertNotIn("current",payload)
        self.assertNotIn("search_begin_input",payload["actor_observation"]);self.assertNotIn("step",payload["actor_observation"]);self.assertNotIn("remainingOverageTime",payload["actor_observation"])
        entity=payload["actor_observation"]["current"]["players"][0]["active"][0];self.assertEqual(entity["energies"],[101,102]);self.assertEqual(entity["energyCards"][0]["id"],201);self.assertEqual(entity["preEvolution"][0]["id"],301);self.assertEqual(entity["tools"][0]["id"],401)
    def test_raw_active_fallback_uses_next_action(self):
        e=episode(visual=False);record=list(dataset.iter_records(e,manifest(e),frozen_unknown_option_fields=frozenset()))[0];self.assertEqual(record.ordered_action,(0,))
        e=episode(visual=False);e.payload["steps"][1][0]["observation"]["current"]["yourIndex"]=1
        with self.assertRaisesRegex(ValueError,"actor perspective"):list(dataset.iter_records(e,manifest(e),frozen_unknown_option_fields=frozenset()))
    def test_null_unauthorized_looking_is_empty_identity_view(self):
        e=episode();e.payload["steps"][0][0]["visualize"][1]["obs"]["current"]["looking"]=None;e.payload["steps"][1][0]["observation"]["current"]["looking"]=None
        record=list(dataset.iter_records(e,manifest(e),frozen_unknown_option_fields=frozenset()))[0]
        self.assertEqual(record.actor_observation["current"]["looking"],())
    def test_null_hidden_opponent_hand_is_empty_identity_view(self):
        e=episode();e.payload["steps"][0][0]["visualize"][1]["obs"]["current"]["players"][1]["hand"]=None;e.payload["steps"][1][0]["observation"]["current"]["players"][1]["hand"]=None
        record=list(dataset.iter_records(e,manifest(e),frozen_unknown_option_fields=frozenset()))[0]
        self.assertEqual(record.actor_observation["current"]["players"][1]["hand"],())
    def test_rejects_hidden_opponent_private_zone_entities(self):
        e=episode();hidden={"id":999,"serial":123};e.payload["steps"][0][0]["visualize"][1]["obs"]["current"]["players"][1]["hand"]=[hidden];e.payload["steps"][1][0]["observation"]["current"]["players"][1]["hand"]=[hidden];e.payload["steps"][0][0]["visualize"][1]["obs"]["current"]["players"][1]["handCount"]=1;e.payload["steps"][1][0]["observation"]["current"]["players"][1]["handCount"]=1
        with self.assertRaisesRegex(ValueError,"opponent private zone"):list(dataset.iter_records(e,manifest(e),frozen_unknown_option_fields=frozenset()))
    def test_visual_cross_checks_and_unique_winner(self):
        e=episode();e.payload["steps"][2][0]["action"]=[1]
        with self.assertRaisesRegex(ValueError,"shifted action"):list(dataset.iter_records(e,manifest(e),frozen_unknown_option_fields=frozenset()))
        e=episode();e.payload["rewards"]=[1,1]
        with self.assertRaisesRegex(ValueError,"unique completed"):list(dataset.iter_records(e,manifest(e),frozen_unknown_option_fields=frozenset()))
    def test_multiset_deck_and_unknown_option_gate(self):
        self.assertEqual(dataset.DeckManifest.from_card_ids([1]*30+[2]*30).sha256,dataset.DeckManifest.from_card_ids([2,1]*30).sha256)
        e=episode(unknown=True)
        with self.assertRaisesRegex(ValueError,"unaudited fields|unknown option"):list(dataset.iter_records(e,manifest(e),frozen_unknown_option_fields=frozenset()))
    def test_assignment_commitment_lookup(self):
        e=episode();m=manifest(e);deck=dataset.DeckManifest.from_card_ids(e.payload["steps"][0][0]["visualize"][0]["action"][0]);self.assertIn(m.assignment_for(split.SplitGroup("2026-07-24",deck.sha256,81,0)).split,{"train","validation"});bad=split.SplitManifest(m.audit,m.assignments,{})
        with self.assertRaisesRegex(ValueError,"commitment"):bad.assignment_for(split.SplitGroup("2026-07-24",deck.sha256,81,0))
    def _build(self,root):
        episodes=(episode(episode_id=81),episode(episode_id=82));m=manifest(episodes[0]);writer=shards.AtomicShardWriter(root,shard_size=1)
        for e in episodes:
            for record in dataset.iter_records(e,m,frozen_unknown_option_fields=frozenset()):writer.write(record.split,record)
        return writer.finalize({},metadata(m))
    def test_deterministic_prevalidated_reference_reader(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/"a";b=Path(d)/"b";ra=self._build(a);rb=self._build(b);self.assertEqual([x["sha256"] for v in ra["shards"].values() for x in v],[x["sha256"] for v in rb["shards"].values() for x in v]);last=[x for v in ra["shards"].values() for x in v][-1];(a/last["path"]).write_bytes((a/last["path"]).read_bytes()+b"tamper")
            with self.assertRaisesRegex(ValueError,"SHA-256 mismatch"):next(shards.iter_dataset(a))
    def test_explicit_unknown_serial_metadata_is_audited_and_inert(self):
        e=episode(unknown=True);audit=Counter();record=list(dataset.iter_records(e,manifest(e),frozen_unknown_option_fields=frozenset({"mystery"}),audit=audit))[0]
        self.assertEqual(audit,{"mystery":1});self.assertNotIn("mystery",record.actor_observation["select"]["option"][0])
    def test_fallback_registration_uses_step_one_decks(self):
        e=episode(visual=False);self.assertEqual(dataset.DeckManifest.from_card_ids(e.payload["steps"][1][0]["action"]).sha256,dataset.DeckManifest.from_card_ids([1]*30+[2]*30).sha256)
    def test_from_dict_rejects_nonwinning_outcome_and_bad_source_hash(self):
        e=episode();r=list(dataset.iter_records(e,manifest(e),frozen_unknown_option_fields=frozenset()))[0];payload=r.as_dict();payload["terminal_outcome"]="loss"
        with self.assertRaisesRegex(ValueError,"winner-only"):dataset.DecisionRecord.from_dict(payload)
        payload=r.as_dict();payload["source_payload_sha256"]="z"*64
        with self.assertRaisesRegex(ValueError,"SHA-256"):dataset.DecisionRecord.from_dict(payload)
    def test_finalize_failure_rolls_back_owned_publication(self):
        e=episode();m=manifest(e)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"failure";writer=shards.AtomicShardWriter(root,shard_size=1)
            for source_episode in (episode(episode_id=81),episode(episode_id=82)):
                for r in dataset.iter_records(source_episode,m,frozen_unknown_option_fields=frozenset()):writer.write(r.split,r)
            with patch.object(shards,"_fsync_dir",side_effect=[None,None,OSError("injected")]):
                with self.assertRaisesRegex(OSError,"injected"):writer.finalize({},metadata(m))
            self.assertFalse(root.exists())
    def test_publication_owner_is_deterministic_tamper_evident_and_directories_fail(self):
        with tempfile.TemporaryDirectory() as d:
            first=Path(d)/"first";second=Path(d)/"second";self._build(first);self._build(second)
            self.assertEqual((first/"publication_owner.json").read_bytes(),(second/"publication_owner.json").read_bytes())
            owner_path=first/"publication_owner.json";owner=json.loads(owner_path.read_text());owner["dataset_content_sha256"]="0"*64;owner_path.write_text(json.dumps(owner))
            with self.assertRaisesRegex(ValueError,"owner|SHA-256"):next(shards.iter_dataset(first))
            (second/"undeclared_directory").mkdir()
            with self.assertRaisesRegex(ValueError,"undeclared"):next(shards.iter_dataset(second))
    def test_build_binds_declared_source_and_protocol_hashes(self):
        e=episode();m=manifest(e)
        with tempfile.TemporaryDirectory() as d:
            with patch.object(dataset,"iter_canonical_episodes") as reader:
                reader.side_effect=lambda sources,sink:(sink((e,)),source.SourceAudit(seen=1,eligible=1))[1]
                with self.assertRaisesRegex(ValueError,"source manifest SHA-256"):
                    dataset.build_dataset([],Path(d)/"bad_source",split_manifest=m,source_manifest_sha256="x",protocol_sha256=PROTOCOL.sha256(),frozen_unknown_option_fields=frozenset())
                with self.assertRaisesRegex(ValueError,"protocol SHA-256"):
                    dataset.build_dataset([],Path(d)/"bad_protocol",split_manifest=m,source_manifest_sha256=dataset.SOURCE_MANIFEST_SHA256,protocol_sha256="b"*64,frozen_unknown_option_fields=frozenset())
    def test_auto_build_derives_group_split_in_one_transaction(self):
        episodes=(episode(episode_id=81),episode(episode_id=82));calls=[]
        with tempfile.TemporaryDirectory() as d,patch.object(dataset,"iter_canonical_episodes") as reader:
            def feed(sources,sink):
                calls.append("scan");sink(episodes);return source.SourceAudit(seen=2,eligible=2)
            reader.side_effect=feed
            reference,private=dataset.build_dataset_with_derived_split([],Path(d)/"dataset",source_manifest_sha256=dataset.SOURCE_MANIFEST_SHA256,protocol=PROTOCOL,frozen_unknown_option_fields=frozenset())
            self.assertEqual(calls,["scan"])
            self.assertEqual(reference["counts"]["train"]+reference["counts"]["validation"],2)
            self.assertEqual(private.audit.private_assignments_count,2)
    def test_required_distribution_audit_is_published(self):
        episodes=(episode(episode_id=81),episode(episode_id=82));m=manifest(episodes[0])
        with tempfile.TemporaryDirectory() as d,patch.object(dataset,"iter_canonical_episodes") as reader:
            reader.side_effect=lambda sources,sink:(sink(episodes),source.SourceAudit(seen=2,eligible=2))[1]
            reference=dataset.build_dataset([],Path(d)/"dataset",split_manifest=m,source_manifest_sha256=dataset.SOURCE_MANIFEST_SHA256,protocol_sha256=PROTOCOL.sha256(),frozen_unknown_option_fields=frozenset())
            audit=reference["distribution_audit"]
            self.assertEqual(audit["group_counts_by_date"],{"2026-07-24":2})
            self.assertEqual(sum(audit["decision_counts_by_action_type"].values()),2)
            self.assertEqual(sum(audit["decision_counts_by_select_type"].values()),2)
            self.assertEqual(audit["cross_stratum_conflicts"],{"comparable_state_groups":0,"conflicting_groups":0,"conflict_rate":0.0})
    def test_reference_types_and_abort_foreign_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"dataset";self._build(root);path=root/"dataset_reference.json";ref=json.loads(path.read_text());ref["counts"]["train"]=True;path.write_text(json.dumps(ref))
            with self.assertRaisesRegex(ValueError,"reference"):next(shards.iter_dataset(root))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"dataset";writer=shards.AtomicShardWriter(root,shard_size=1);writer._release_reservation();root.mkdir();(root/"foreign").write_text("keep");writer.abort();self.assertEqual((root/"foreign").read_text(),"keep")
if __name__=="__main__":unittest.main()
