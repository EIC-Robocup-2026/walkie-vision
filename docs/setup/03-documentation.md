# 3. Add the Documentation Site

**Goal:** build this website from Markdown files in the `docs/` folder, and
publish it to GitHub Pages automatically.

We use [MkDocs](https://www.mkdocs.org/) with the
[Material](https://squidfunk.github.io/mkdocs-material/) theme, the same as
walkie-agent-ros and walkie-database. You write normal Markdown, and MkDocs
turns it into a website.

## Step 1: Install MkDocs in its own environment

```toml title="pixi.toml"
[feature.docs.dependencies]
mkdocs-material = ">=9.7,<10"
mkdocs = ">=1.6,<2"

[feature.docs.activation.env]
RMW_IMPLEMENTATION = "rmw_zenoh_cpp"

[environments]
docs = { features = ["docs"], no-default-feature = true }
```

What these do:

- A **feature** is a named group of packages. The `docs` feature holds
  MkDocs.
- An **environment** is a set of features that is installed together. The
  `docs` environment holds only the `docs` feature.
- `no-default-feature = true` keeps ROS, PyTorch and the CUDA libraries out
  of the `docs` environment. It installs in seconds instead of downloading
  about 9.6 GB, which matters on GitHub Actions.
- The default feature's `[activation.env]` does not apply to the `docs`
  environment either, so the `docs` feature sets the same middleware. Every
  environment then agrees, although MkDocs never uses it.

!!! info "Why `mkdocs<2`?"
    MkDocs 2.0 removes the plugin system, so the Material theme does not work
    with it. The version limit makes sure Pixi never installs it by accident.
    `mkdocs build` prints a warning about MkDocs 2.0. It is not an error.

The `docs` environment has no Python pin, so it uses the newest Python that
MkDocs supports (3.15 in `pixi.lock`), not the 3.12 of the default
environment.

## Step 2: Add tasks

```toml title="pixi.toml"
[feature.docs.tasks]
docs = "mkdocs serve"
docs-build = "mkdocs build --strict"
```

| Command               | What it does                                                                                       |
| --------------------- | -------------------------------------------------------------------------------------------------- |
| `pixi run docs`       | Preview the site at <http://127.0.0.1:8000>. Reloads when you save.                                |
| `pixi run docs-build` | Build the final site into `site/`. `--strict` makes it fail on any warning, such as a broken link. |

These tasks only exist in the `docs` environment, so Pixi picks that
environment automatically.

## Step 3: Configure the site

`mkdocs.yml` in the project root controls the site. It is copied from
walkie-database, with the names and the menu changed. The important parts:

```yaml title="mkdocs.yml (part)"
site_name: Walkie Vision # title shown at the top

theme:
  name: material # use the Material theme

# Links that start with "/" are resolved from docs/.
validation:
  links:
    absolute_links: relative_to_docs

# ADRs are embedded in the architecture pages instead of listed in the menu.
not_in_nav: |
  architecture/decisions/*

nav: # the menu on the left
  - Home: index.md
  - Getting Started: getting-started.md
```

The rest of the file turns on extra features, like the dark mode toggle,
copy buttons on code blocks, Mermaid diagrams, and the coloured note boxes
you see on this page. `pymdownx.snippets` embeds the
[decision records](../architecture/index.md) into the architecture pages,
and fails the build when an embedded file or section is missing.

Git ignores `site/`, because it is build output.

## Step 4: Publish with GitHub Actions

The file `.github/workflows/docs.yml` runs on GitHub every time someone
pushes changes to `docs/`, `mkdocs.yml`, `pixi.toml`, `pixi.lock` or the
workflow itself on `main`. It:

1. Installs Pixi and only the `docs` environment, from `pixi.lock`.
2. Runs `pixi run -e docs docs-build`.
3. Uploads `site/` to GitHub Pages.

!!! warning "One-time setting on GitHub"
    In the repository, go to **Settings → Pages** and set **Source** to
    **GitHub Actions**. Without this, the publish step fails.

There is no workflow that runs the tests yet: it would need the default
environment, about 9.6 GB, on every run.

## Writing a new page

1. Create a Markdown file in `docs/`, for example `docs/guides/my-topic.md`.
2. Add it to `nav` in `mkdocs.yml`:

    ```yaml
    nav:
      - Guides:
          - My Topic: guides/my-topic.md
    ```

3. Run `pixi run docs` and check it in your browser.
4. Run `pixi run docs-build` before you commit. If it fails, fix the warning
   it prints.

Some useful things you can write in a page:

=== "Note box"

    ```markdown
    !!! note "Title"
        Text inside the box. Indent it with 4 spaces.
    ```

    Other box types: `tip`, `info`, `warning`, `danger`.

=== "Code block with title"

    ````markdown
    ```python title="detect.py"
    results = model.predict(image, quantize=16)
    ```
    ````

=== "Tabs"

    ```markdown
    === "Tab one"

        Content of tab one.

    === "Tab two"

        Content of tab two.
    ```

=== "Diagram"

    ````markdown
    ```mermaid
    flowchart LR
      camera --> torch[torch process] --> db[walkie-database]
    ```
    ````

=== "Image"

    Put image files in `docs/assets/`, then link them with a path relative
    to your page:

    ```markdown
    ![detections on the table](../assets/detections.png){ width="500" }
    ```

    From `docs/setup/my-page.md` the path starts with `../assets/`. From a
    page directly in `docs/`, it is `assets/detections.png`.

    Use PNG for screenshots and keep files under about 1 MB, because every
    image is stored in Git forever.

See the [Material reference](https://squidfunk.github.io/mkdocs-material/reference/)
for everything else.

To record a decision, see [Add a decision](../architecture/index.md#add-a-decision).
