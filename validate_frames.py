import json, pathlib
from openpi.training import robomind as rm
recs=json.load(open("data/robomind_records_bread.json"))
labs=json.load(open("data/robomind_labels_bread.json"))
# mark the partial extract as complete so fetch_task_hdf5 globs it (no re-extract/download)
done=pathlib.Path("data/robomind_cache_bread/bread_in_basket/.done"); done.parent.mkdir(parents=True,exist_ok=True); done.touch()
rows=rm.assemble(recs, labs, out_dir="data/robomind_hl_bread", cache_dir="data/robomind_cache_bread",
                 fps_default=10.0, sample_hz=1.0, max_episodes=10)
print("MANIFEST_ROWS=",len(rows))
for r in rows[:8]:
    print("  frame=%s update=%s tgt_subtask=%r tgt_mem=%r img=%s"%(
        r["frame"], r["update"], r["target_subtask"], r["target_memory"], r["image"]))
