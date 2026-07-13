# E1 ExecuTorch Raspberry Pi Baseline Mutation And Rollback Manifest

Created: 2026-07-13
Scope: E1-only ExecuTorch baseline bring-up.

## Non-Project Workspace

GPU Linux workspace:
/home/allen/executorch_e1/
subdirectories: src, artifacts, logs, manifests, export, runner

Raspberry Pi workspace:
/home/allen/executorch_e1/

No Compiler, Runtime, Capability DB, IVP, P1D deployment, target profile, or production source directories are modified by this plan.

## Planned Source Changes

None to project production repositories.

## Planned Repositories To Clone

Repository: https://github.com/pytorch/executorch.git
Pinned tag: v1.3.1
Resolved commit: e2f18eb23c45bd22ca332b0b8b49a81de304b472
Clone location: /home/allen/executorch_e1/src/executorch

## Planned Package / Tool Mutations

Preferred first pass avoids system package installation.

If build fails due missing tools, possible host-only packages may be required later and must be recorded before execution:
- GPU Linux: ninja-build may be needed for CMake/Ninja builds.
- Raspberry Pi: cmake and possibly clang may be needed only for native fallback build.

No Pi package installation is performed until required and separately recorded in E1 logs.

## Python Environments

Create user-local virtual environments only under /home/allen/executorch_e1/.
Do not modify system Python.
Do not install into project virtual environments.
Do not install global Python packages.

## Build Outputs

All generated ExecuTorch source/build/export/deployment artifacts remain under /home/allen/executorch_e1/ unless explicitly copied to the Pi E1 workspace.

Do not commit:
- ExecuTorch source checkout
- build trees
- object files
- third-party generated libraries
- virtual environments
- large pte or binary artifacts without explicit policy review

## Pi Deployment Files

Copy only into /home/allen/executorch_e1/:
- runner binary
- required shared libraries if any
- pte artifacts
- input/reference data or deterministic workload manifest
- provenance/checksum manifests
- smoke results

## Environment Variables

Set only per-shell/build-command variables. No persistent shell startup files are modified.

## System Files

No system files, boot config, governor config, or system XNNPACK libraries are changed.

## Expected Disk Use

Estimated:
- ExecuTorch source plus submodules: 2-6 GB depending on submodules.
- Build trees: 2-10 GB.
- E1 Pi deployment artifacts: under 500 MB expected.

Current observed free space:
- GPU /home: about 55 GB free.
- Pi /home: about 20 GB free.

## Rollback Commands

GPU Linux:
rm -rf /home/allen/executorch_e1

Raspberry Pi:
rm -rf /home/allen/executorch_e1

If any system packages are installed later, record exact package names and install logs before installation; rollback for packages requires explicit review and is not part of the default plan.

## Safety Boundaries

- No project production code changes.
- No project kernels changed.
- No P1D evidence changed.
- No ExecuTorch custom operator wrapping the project kernel.
- No formal head-to-head campaign.
- No superiority claim.
