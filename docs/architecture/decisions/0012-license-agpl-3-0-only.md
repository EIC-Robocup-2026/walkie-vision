# ADR-0012: License the repository under AGPL-3.0-only

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-28

**Context**

The repository is public on GitHub, but it had no licence, so nobody had
permission to reuse its code.

The vision node depends on Ultralytics (`ultralytics >=8.4.182,<8.5` in
`pixi.toml`) for YOLOE-26 and YOLO26. Ultralytics is AGPL-3.0. Its 8.4.182
release states the version in several places:

- the `License` field of its package metadata says `AGPL-3.0`;
- its `LICENSE` is the plain AGPL-3.0 text from gnu.org;
- every `.py` file starts with `# Ultralytics 🚀 AGPL-3.0 License`, and none
  of them says "or any later version";
- only its trove classifier says "v3 or later (AGPLv3+)".

The models have licences of their own, which a licence for the code cannot
change. `buffalo_l` and `mobileclip2_b.ts` are for non-commercial research
only ([Licences](/setup/01-vision-environment.md#licences)).

walkie-database and walkie-agent-ros build `walkie_vision_msgs` in their own
workspaces ([ADR-0003](/architecture/decisions/0003-messages-in-this-repository.md)).

**Decision**

- The team decided on 2026-10-28 to license the repository under AGPL-3.0,
  to match Ultralytics.
- The SPDX identifier is `AGPL-3.0-only`: version 3, without "or any later
  version", as Ultralytics' file headers state it.
- `LICENSE` at the repository root is the unmodified AGPL-3.0 text from
  gnu.org, the same file that Ultralytics 8.4.182 ships.
- `pyproject.toml` (`license`) and the message package's `package.xml`
  (`<license>`) say `AGPL-3.0-only`. The message package has no licence of
  its own.
- The vendored files in `src/walkie_vision/face/` and
  `src/walkie_vision/reid/` keep their MIT licence headers.
- The models keep their own licences. `LICENSE` does not cover them.

**Consequences**

- Anyone may use, change and share the code under AGPL-3.0. If you give
  someone a copy, they must be able to get its source (sections 4 to 6). If
  you modify it and let people use it over a network, you must offer them
  its source (section 13). Keeping the code that runs on the robot pushed
  to this repository makes its source available. If you modify it and let
  others use it over a network, or give someone a copy, also point them to
  this repository.
- A change of licence needs the agreement of everyone who holds copyright
  in the team's own code. The vendored MIT files do not need it, because
  MIT allows sublicensing. Moving to "or any later version" stays possible
  later. Once a version is released under "or any later version", that
  grant cannot be withdrawn for that version.
- walkie-database and walkie-agent-ros build `walkie_vision_msgs`, so their
  workspaces contain AGPL-3.0-only interface files. If they must stay clear
  of AGPL, the message package can get a licence of its own in a new ADR.
- `buffalo_l` and `mobileclip2_b.ts` stay non-commercial research only,
  whatever the code's licence. The team uses them for RoboCup, a research
  competition.
- `pyproject.toml` gives the licence as an SPDX string, which needs
  setuptools 77 or newer. `[build-system]` requires it.

**See also:** [Licences](/setup/01-vision-environment.md#licences),
[Add the Message Package](/setup/02-message-package.md)
<!-- --8<-- [end:body] -->
