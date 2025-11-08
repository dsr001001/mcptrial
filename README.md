# gimpmcpr1

Automate GIMP scenes via the Model Control Protocol (MCP). The project reads
plain-text scene descriptions and expands them into MCP command plans that can
be executed by a running GIMP instance.

## Quick start

```bash
python scripts/render_scene.py instructions.txt --output-dir output
```

Enable the `--send` flag to stream the generated plan directly to a MCP daemon
(defaults to `127.0.0.1:10002`).

## Instruction file format

Instructions are defined in `instructions.txt`. Each scene begins with a
`scene:` header and can contain `canvas`, `background`, and one or more
`element` blocks separated by `---` lines. Elements support lightweight
attributes for layout, palette, and style. For example:

```
scene: sunny-home
canvas: 1280x720
background: sky-blue
---
element: house size=medium position=center palette=warm
layer: foreground
---
element: sun size=small position=top-right rays=12 glow=true
---
element: garden type=flowerbed band=foreground density=medium colors=pink,orange,yellow
```

## Output

Running the renderer writes one JSON plan per scene into `output/`. Each plan
captures the resolved scene metadata and the ordered list of MCP commands.

## Extending the vocabulary

Scene element handling is implemented in `scripts/render_scene.py`. Extend the
`CommandBuilder` class with new `handle_<element>()` methods to introduce more
building blocks (e.g., clouds, mountains, trees). Each handler should return a
list of command definitions structured for the target MCP endpoint.

## MCP connectivity

The bundled `GimpMcpClient` streams the plan over a simple TCP socket where the
daemon is expected to respond with `OK`. Adjust the host/port via CLI flags or
adapt the client to match your deployment. If no daemon is available, omit the
`--send` flag and import the JSON plan manually inside GIMP.
