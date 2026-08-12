from __future__ import annotations

import json

import wandb


def main() -> None:
    run = wandb.init(
        project="pokemon-cuda-engine-smoke",
        name="install-validation",
        mode="offline",
        config={"kind": "offline_install_smoke"},
    )
    run.log({"engine/smoke": 1, "gpu/vram_mib": 32228.812})
    run.finish()
    print(
        json.dumps(
            {
                "wandb_version": wandb.__version__,
                "mode": "offline",
                "run_id": run.id,
                "run_dir": run.dir,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
