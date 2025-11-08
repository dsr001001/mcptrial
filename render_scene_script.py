"""Render scenes in GIMP via MCP based on declarative instructions.

The script ingests a lightweight instruction file that describes one or
more scenes using a compact text format. Each scene is expanded into a
sequence of GIMP MCP commands and written to `output/<scene>.json`.

If a MCP daemon is reachable, the plan can optionally be streamed to it
so the artwork is generated automatically inside GIMP.

Example usage:

    python scripts/render_scene.py instructions.txt --send

"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data model


@dataclass
class SceneElement:
    element_type: str
    attributes: Dict[str, str]
    layer: Optional[str] = None


@dataclass
class SceneInstruction:
    name: str
    canvas: Tuple[int, int]
    background: str
    raw_blocks: List[List[str]] = field(default_factory=list)
    elements: List[SceneElement] = field(default_factory=list)

    def to_serialisable(self) -> Dict[str, object]:
        return {
            "scene": self.name,
            "canvas": {"width": self.canvas[0], "height": self.canvas[1]},
            "background": self.background,
            "elements": [
                {
                    "type": element.element_type,
                    "layer": element.layer,
                    "attributes": element.attributes,
                }
                for element in self.elements
            ],
        }


@dataclass
class Command:
    action: str
    params: Dict[str, object]
    comment: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        payload = {"action": self.action, "params": self.params}
        if self.comment:
            payload["comment"] = self.comment
        return payload


# ---------------------------------------------------------------------------
# Instruction parsing


class InstructionParser:
    """Parse the domain-specific instruction file into structured scenes."""

    def __init__(self, default_canvas: Tuple[int, int] = (1024, 768)) -> None:
        self.default_canvas = default_canvas

    def parse(self, path: Path) -> List[SceneInstruction]:
        lines = [line.rstrip() for line in path.read_text().splitlines()]
        scenes: List[SceneInstruction] = []

        current_scene: Optional[SceneInstruction] = None
        buffer: List[str] = []

        def flush_block() -> None:
            nonlocal buffer, current_scene
            if current_scene is None or not buffer:
                buffer = []
                return
            current_scene.raw_blocks.append(buffer)
            buffer = []

        for line in lines + [""]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                flush_block()
                continue

            if stripped.lower().startswith("scene:"):
                flush_block()
                if current_scene:
                    self._finalise_scene(current_scene)
                    scenes.append(current_scene)
                name = stripped.split(":", 1)[1].strip()
                current_scene = SceneInstruction(
                    name=name,
                    canvas=self.default_canvas,
                    background="sky-blue",
                )
                continue

            if stripped == "---":
                flush_block()
                continue

            buffer.append(stripped)

        if current_scene:
            flush_block()
            self._finalise_scene(current_scene)
            scenes.append(current_scene)

        if not scenes:
            raise ValueError(f"No scenes found in {path}")

        return scenes

    def _finalise_scene(self, scene: SceneInstruction) -> None:
        for block in scene.raw_blocks:
            if not block:
                continue

            header = block[0]
            key, _, value = header.partition(":")
            key = key.strip().lower()
            value = value.strip()

            if key == "canvas":
                scene.canvas = self._parse_canvas(value)
                continue

            if key == "background":
                scene.background = value
                continue

            if key == "element":
                element_type, attrs = self._parse_element(value, block[1:])
                scene.elements.append(
                    SceneElement(
                        element_type=element_type,
                        attributes=attrs["attributes"],
                        layer=attrs.get("layer"),
                    )
                )
                continue

            raise ValueError(f"Unknown directive '{key}' in scene '{scene.name}'")

    @staticmethod
    def _parse_canvas(value: str) -> Tuple[int, int]:
        parts = value.lower().replace("x", " ").split()
        if len(parts) != 2:
            raise ValueError(f"Invalid canvas format '{value}' (expected WxH)")
        width, height = (int(parts[0]), int(parts[1]))
        if width <= 0 or height <= 0:
            raise ValueError(f"Canvas dimensions must be positive: '{value}'")
        return width, height

    @staticmethod
    def _parse_element(value: str, extra_lines: Iterable[str]) -> Tuple[str, Dict[str, Dict[str, str]]]:
        tokens = value.split()
        if not tokens:
            raise ValueError("Element directive requires a type")
        element_type = tokens[0].lower()
        attributes: Dict[str, str] = {}

        for token in tokens[1:]:
            if "=" not in token:
                continue
            key, val = token.split("=", 1)
            attributes[key.lower()] = val

        layer: Optional[str] = None
        for line in extra_lines:
            key, _, val = line.partition(":")
            key = key.strip().lower()
            val = val.strip()
            if key == "layer":
                layer = val
            else:
                attributes[key] = val

        return element_type, {"attributes": attributes, "layer": layer}


# ---------------------------------------------------------------------------
# Rendering plan builders


class CommandBuilder:
    """Expand high-level scene elements into MCP command dictionaries."""

    SIZE_FACTORS = {"tiny": 0.08, "small": 0.16, "medium": 0.28, "large": 0.4}
    POSITIONS = {
        "center": (0.5, 0.5),
        "center-left": (0.35, 0.5),
        "center-right": (0.65, 0.5),
        "top": (0.5, 0.2),
        "top-left": (0.3, 0.18),
        "top-right": (0.7, 0.18),
        "bottom": (0.5, 0.75),
        "bottom-left": (0.3, 0.75),
        "bottom-right": (0.7, 0.75),
        "foreground": (0.5, 0.85),
    }

    PALETTES = {
        "warm": ["#f4a261", "#e76f51", "#fb8500"],
        "cool": ["#219ebc", "#8ecae6", "#023047"],
        "garden": ["#2a9d8f", "#52b788", "#95d5b2"],
        "twilight": ["#6d597a", "#b56576", "#355070"],
    }

    def __init__(self, scene: SceneInstruction) -> None:
        self.scene = scene
        self.width, self.height = scene.canvas

    def build_plan(self) -> List[Command]:
        commands: List[Command] = []

        commands.append(
            Command(
                action="ensure-canvas",
                params={"width": self.width, "height": self.height},
                comment="Create or resize the canvas",
            )
        )
        commands.append(
            Command(
                action="set-background",
                params={"style": self.scene.background},
                comment="Scene background",
            )
        )

        for idx, element in enumerate(self.scene.elements, start=1):
            handler = getattr(self, f"handle_{element.element_type}", None)
            if handler is None:
                commands.append(
                    Command(
                        action="annotate",
                        params={
                            "message": f"TODO: element '{element.element_type}' not implemented",
                            "attributes": element.attributes,
                        },
                        comment=f"Placeholder for element {idx}",
                    )
                )
                continue
            commands.extend(handler(element))

        return commands

    def resolve_position(self, tag: str) -> Tuple[float, float]:
        return self.POSITIONS.get(tag.lower(), self.POSITIONS["center"])

    def resolve_size(self, tag: str) -> float:
        return self.SIZE_FACTORS.get(tag.lower(), self.SIZE_FACTORS["medium"])

    def resolve_palette(self, key: str, fallback: str = "warm") -> List[str]:
        return self.PALETTES.get(key.lower(), self.PALETTES.get(fallback, ["#cccccc"]))

    # -- Element handlers -------------------------------------------------

    def handle_house(self, element: SceneElement) -> List[Command]:
        size = self.resolve_size(element.attributes.get("size", "medium"))
        palette_key = element.attributes.get("palette", "warm")
        palette = self.resolve_palette(palette_key)
        pos = self.resolve_position(element.attributes.get("position", "center"))
        width = int(self.width * size)
        height = int(self.height * size * 0.8)

        cx = int(self.width * pos[0])
        cy = int(self.height * pos[1])
        layer_name = element.layer or "house"

        return [
            Command(
                action="new-layer",
                params={"name": layer_name, "mode": "normal", "opacity": 100},
                comment="Layer for house body",
            ),
            Command(
                action="draw-rectangle",
                params={
                    "layer": layer_name,
                    "x": cx - width // 2,
                    "y": cy,
                    "width": width,
                    "height": height,
                    "fill": palette[0],
                    "border": {"width": 4, "color": palette[-1]},
                },
                comment="House body",
            ),
            Command(
                action="draw-triangle",
                params={
                    "layer": layer_name,
                    "points": self._roof_points(cx, cy, width, height // 2),
                    "fill": palette[1],
                    "border": {"width": 3, "color": palette[-1]},
                },
                comment="Roof",
            ),
            Command(
                action="draw-rectangle",
                params={
                    "layer": layer_name,
                    "x": cx - width // 8,
                    "y": cy + height // 2,
                    "width": width // 4,
                    "height": height // 2,
                    "fill": "#ffffff",
                    "border": {"width": 2, "color": palette[-1]},
                },
                comment="Door",
            ),
        ]

    def handle_sun(self, element: SceneElement) -> List[Command]:
        size = self.resolve_size(element.attributes.get("size", "small")) * 0.8
        pos = self.resolve_position(element.attributes.get("position", "top-right"))
        radius = int(min(self.width, self.height) * size / 2)
        cx = int(self.width * pos[0])
        cy = int(self.height * pos[1])
        rays = int(element.attributes.get("rays", 12))
        glow = element.attributes.get("glow", "false").lower() == "true"
        palette = self.resolve_palette("warm")

        commands = [
            Command(
                action="new-layer",
                params={"name": "sun", "mode": "add", "opacity": 100},
                comment="Sun layer",
            ),
            Command(
                action="draw-circle",
                params={
                    "layer": "sun",
                    "center": [cx, cy],
                    "radius": radius,
                    "fill": palette[0],
                    "border": {"width": 3, "color": palette[-1]},
                },
                comment="Solar disk",
            ),
        ]

        commands.append(
            Command(
                action="draw-rays",
                params={
                    "layer": "sun",
                    "center": [cx, cy],
                    "count": rays,
                    "inner_radius": radius + 4,
                    "outer_radius": radius + radius // 2,
                    "color": palette[1],
                },
                comment="Sun rays",
            )
        )

        if glow:
            commands.append(
                Command(
                    action="apply-glow",
                    params={
                        "layer": "sun",
                        "radius": radius // 2,
                        "strength": 0.6,
                        "color": palette[1],
                    },
                    comment="Sun glow",
                )
            )

        return commands

    def handle_garden(self, element: SceneElement) -> List[Command]:
        density = element.attributes.get("density", "medium")
        band = element.attributes.get("band", "foreground")
        colors = [c.strip() for c in element.attributes.get("colors", "").split(",") if c.strip()]
        palette = colors or self.resolve_palette("garden")
        randomness = float(element.attributes.get("randomness", 0.25))

        return [
            Command(
                action="scatter-brush",
                params={
                    "layer": element.layer or "garden",
                    "band": band,
                    "density": density,
                    "colors": palette,
                    "randomness": randomness,
                    "brush": element.attributes.get("brush", "foliage"),
                },
                comment="Garden foliage",
            )
        ]

    def handle_skyline(self, element: SceneElement) -> List[Command]:
        towers = int(element.attributes.get("towers", 5))
        variation = element.attributes.get("variation", "moderate")
        lights = element.attributes.get("lights", "off").lower() == "on"

        return [
            Command(
                action="draw-skyline",
                params={
                    "layer": element.layer or "skyline",
                    "towers": towers,
                    "variation": variation,
                    "lights": lights,
                    "palette": self.resolve_palette("twilight"),
                },
                comment="Skyline silhouette",
            )
        ]

    def handle_moon(self, element: SceneElement) -> List[Command]:
        phase = element.attributes.get("phase", "crescent")
        pos = self.resolve_position(element.attributes.get("position", "top-right"))
        glow = element.attributes.get("glow", "none")

        return [
            Command(
                action="draw-moon",
                params={
                    "layer": element.layer or "moon",
                    "phase": phase,
                    "center": [int(self.width * pos[0]), int(self.height * pos[1])],
                    "radius": int(min(self.width, self.height) * 0.08),
                    "glow": glow,
                },
                comment="Moon",
            )
        ]

    def handle_stars(self, element: SceneElement) -> List[Command]:
        density = element.attributes.get("density", "medium")
        randomness = float(element.attributes.get("randomness", 0.25))

        return [
            Command(
                action="scatter-stars",
                params={
                    "layer": element.layer or "stars",
                    "density": density,
                    "randomness": randomness,
                    "palette": ["#ffffff", "#ffe066", "#f7f1e1"],
                },
                comment="Star field",
            )
        ]

    @staticmethod
    def _roof_points(cx: int, cy: int, width: int, height: int) -> List[List[int]]:
        return [[cx - width // 2, cy], [cx + width // 2, cy], [cx, cy - height]]


# ---------------------------------------------------------------------------
# MCP client (best-effort)


class GimpMcpClient:
    """Minimal TCP client for sending JSON commands to a GIMP MCP daemon.

    The protocol is intentionally simple: each payload is serialized to a
    single JSON line and terminated with a newline. The daemon should reply
    with either `OK` or an error string per line.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 10002, timeout: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

    def send_plan(self, scene: SceneInstruction, commands: List[Command]) -> None:
        payload = {
            "scene": scene.name,
            "commands": [cmd.to_dict() for cmd in commands],
        }
        data = json.dumps(payload).encode("utf-8") + b"\n"

        with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
            sock.sendall(data)
            response = sock.recv(4096).decode("utf-8").strip()
            if response and response.upper() != "OK":
                raise RuntimeError(f"GIMP MCP rejected plan: {response}")


# ---------------------------------------------------------------------------
# CLI


def render(instruction_path: Path, output_dir: Path, *, send: bool = False, host: str = "127.0.0.1", port: int = 10002) -> None:
    parser = InstructionParser()
    scenes = parser.parse(instruction_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = GimpMcpClient(host=host, port=port) if send else None

    for scene in scenes:
        builder = CommandBuilder(scene)
        commands = builder.build_plan()

        json_path = output_dir / f"{scene.name}.json"
        json_path.write_text(
            json.dumps(
                {
                    "scene": scene.to_serialisable(),
                    "commands": [command.to_dict() for command in commands],
                },
                indent=2,
            )
        )

        print(f"Wrote plan for scene '{scene.name}' → {json_path}")

        if client:
            try:
                client.send_plan(scene, commands)
                print(f"Sent plan for '{scene.name}' to MCP at {host}:{port}")
            except Exception as exc:  # broad except to keep loop alive
                print(f"[warn] Failed to deliver plan for '{scene.name}': {exc}", file=sys.stderr)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render GIMP MCP scenes from instructions")
    parser.add_argument("instruction_file", type=Path, help="Path to the instruction text file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for generated JSON plans",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Send the generated plan to a running GIMP MCP daemon",
    )
    parser.add_argument("--host", default="127.0.0.1", help="GIMP MCP host")
    parser.add_argument("--port", type=int, default=10002, help="GIMP MCP port")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        render(args.instruction_file, args.output_dir, send=args.send, host=args.host, port=args.port)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


