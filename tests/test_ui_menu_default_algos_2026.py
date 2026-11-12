"""Regressões de UI sem dependência de pygame no CI.

Os testes inspecionam a ligação dos componentes reais e executam os
helpers puros de seleção e dimensionamento extraídos via AST. O layout
final ainda requer inspeção visual em Windows, com a fonte instalada.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def module_ast(path):
    return ast.parse((ROOT / path).read_text(encoding="utf-8"))


def public_function(tree, name):
    return next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


def call_names(node):
    return {
        n.func.id if isinstance(n.func, ast.Name) else n.func.attr
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, (ast.Name, ast.Attribute))
    }


class TestMenuButtonGeometry(unittest.TestCase):
    def setUp(self):
        self.tree = module_ast("apps/main_interactive.py")

    def test_menu_buttons_fit_two_columns(self):
        main = public_function(self.tree, "main")
        definitions = {}
        for node in ast.walk(main):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            name = node.targets[0]
            if (isinstance(name, ast.Name)
                    and name.id.startswith("btn_")
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "Button"):
                try:
                    definitions[name.id] = (
                        ast.literal_eval(node.value.args[0]),
                        ast.literal_eval(node.value.args[1]),
                    )
                except (ValueError, TypeError):
                    pass
        start, start_label = definitions["btn_start"]
        logs, logs_label = definitions["btn_logs"]
        mapedit, _ = definitions["btn_mapedit"]
        uwb, _ = definitions["btn_uwbtest"]
        self.assertEqual(start_label, "Iniciar simulação")
        self.assertEqual(logs_label, "Abrir pasta de logs")
        self.assertEqual(start[2:], (220, 44))
        self.assertEqual(logs[2:], (220, 44))
        self.assertEqual((start[0], logs[0]), (mapedit[0], uwb[0]))
        self.assertEqual(start[1], logs[1])
        self.assertLessEqual(logs[0] + logs[2], 1280)

    def test_button_uses_native_font_without_any_surface_scaling(self):
        """Protege contra fonte embaçada causada pelo smoothscale do PR15."""
        source = module_ast("src/ui/botton.py")
        button_cls = next(
            n for n in source.body
            if isinstance(n, ast.ClassDef) and n.name == "Button"
        )
        draw = next(
            n for n in button_cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "draw"
        )
        calls = call_names(draw)
        self.assertIn("render", calls)
        self.assertIn("get_rect", calls)
        self.assertIn("blit", calls)
        self.assertIn("set_clip", calls)
        self.assertIn("get_clip", calls)
        self.assertIn("inflate", calls)
        self.assertNotIn("scale", calls)
        self.assertNotIn("smoothscale", calls)

    def test_button_restores_clipping_and_blits_original_rendered_surface(self):
        """Exercita draw() com fakes, sem instalar pygame no CI."""
        from types import SimpleNamespace

        class FakeRect:
            def __init__(self, value):
                self.x, self.y, self.w, self.h = value

            @property
            def center(self):
                return (self.x + self.w // 2, self.y + self.h // 2)

            def inflate(self, dw, dh):
                return FakeRect((self.x-dw//2, self.y-dh//2,
                                 self.w+dw, self.h+dh))

            def clip(self, other):
                x = max(self.x, other.x)
                y = max(self.y, other.y)
                right = min(self.x+self.w, other.x+other.w)
                bottom = min(self.y+self.h, other.y+other.h)
                return FakeRect((x,y,max(0,right-x),max(0,bottom-y)))

            def collidepoint(self, pos):
                return (self.x <= pos[0] < self.x+self.w
                        and self.y <= pos[1] < self.y+self.h)

        class Image:
            def get_rect(self, *, center):
                return center

        class Font:
            def __init__(self):
                self.image = Image()
            def render(self, text, aa, color):
                if text != "Abrir pasta de logs" or not aa:
                    raise AssertionError("Texto ou antialiasing incorreto")
                return self.image

        class Surface:
            def __init__(self):
                self.clip = FakeRect((0, 0, 1280, 760))
                self.previous_clip = self.clip
                self.events = []
            def get_clip(self):
                return self.clip
            def set_clip(self, rect):
                self.clip = rect
                self.events.append(("clip", rect))
            def blit(self, image, rect):
                self.events.append(("blit", image, self.clip))

        fake_pg = SimpleNamespace(
            Rect=FakeRect, draw=SimpleNamespace(rect=lambda *args, **kwargs: None)
        )
        tree = module_ast("src/ui/botton.py")
        cls = next(n for n in tree.body
                   if isinstance(n, ast.ClassDef) and n.name=="Button")
        namespace = {"pg": fake_pg, "LBL": (40,40,40),
                     "BLACK": (20,20,20)}
        exec(compile(ast.Module(body=[cls], type_ignores=[]),
                     "<native_button>", "exec"), namespace)
        font = Font()
        button = namespace["Button"]((720,520,220,44),
                                     "Abrir pasta de logs", font)
        surface = Surface()
        button.draw(surface)
        self.assertTrue(button.hit((750,540)))
        self.assertFalse(button.hit((950,540)))
        blits = [event for event in surface.events if event[0]=="blit"]
        self.assertEqual(len(blits), 1)
        # A imagem desenhada é a MESMA retornada pelo font.render:
        # nenhuma superfície interpolada ou borrada intermediária.
        self.assertIs(blits[0][1], font.image)
        self.assertGreaterEqual(blits[0][2].x, button.rect.x)
        self.assertLessEqual(
            blits[0][2].x+blits[0][2].w,
            button.rect.x+button.rect.w
        )
        self.assertIs(surface.clip, surface.previous_clip)



class TestAlgorithmDefaults(unittest.TestCase):
    def setUp(self):
        self.host = module_ast("src/ui/algo_test_screen.py")
        self.shared = module_ast("src/ui/algo_modes/shared.py")
        klass = next(n for n in self.host.body
                     if isinstance(n, ast.ClassDef) and n.name == "AlgoTestScreen")
        self.methods = {n.name:n for n in klass.body if isinstance(n, ast.FunctionDef)}
        self.init = self.methods["__init__"]

    def test_lmsp_default_active_in_host_and_variants(self):
        selected = None
        variant_initialization = False
        for n in ast.walk(self.init):
            if (isinstance(n, ast.Assign) and len(n.targets)==1
                    and isinstance(n.targets[0], ast.Attribute)
                    and n.targets[0].attr == "selected"):
                selected = ast.literal_eval(n.value)
            if (isinstance(n, ast.Assign) and len(n.targets)==1
                    and isinstance(n.targets[0], ast.Attribute)
                    and n.targets[0].attr == "selected_algo_variants"):
                variant_initialization = (
                    "default_algorithm_variant_state" in call_names(n.value)
                )
        self.assertIsNotNone(selected)
        self.assertTrue(variant_initialization)
        self.assertEqual(selected["lmsp"], True)
        self.assertEqual(
            {name for name, value in selected.items() if value},
            {"trilaterate3d", "lms", "gauss_newton", "lmsp"},
        )
        # Extrai funções puras para validar que o estado inicial
        # corresponde à implementação concreta executada no batch.
        dictionary = next(
            n for n in self.shared.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id=="ALGO_VARIANTS"
                    for t in n.targets)
        )
        helpers = [public_function(self.shared, key) for key in (
            "default_algorithm_variant_state", "get_algorithm_variants",
            "algorithm_active_key", "cycle_algorithm_variant",
            "algorithm_button_label"
        )]
        namespace = {"ALGO_VARIANTS": ast.literal_eval(dictionary.value)}
        exec(compile(ast.Module(body=helpers, type_ignores=[]),
                     "<algorithm_variants>", "exec"), namespace)
        state = namespace["default_algorithm_variant_state"]()
        self.assertEqual(state["lmsp"], "lmsp")
        self.assertEqual(namespace["algorithm_active_key"]("lmsp",selected,state),"lmsp")
        self.assertIn("LMS-P atual",namespace["algorithm_button_label"]("lmsp",selected,state))
        selected["lmsp"], state["lmsp"] = namespace["cycle_algorithm_variant"](
            "lmsp",selected,state
        )
        self.assertTrue(selected["lmsp"])
        self.assertEqual(state["lmsp"], "wls_sigma")
        # Off continua acessível ao completar o ciclo de variantes.
        for _ in range(3):  # sigma -> RKF -> Wang17 -> off
            namespace["cycle_algorithm_variant"]("lmsp",selected,state)
        self.assertIsNone(state["lmsp"])
        self.assertFalse(selected["lmsp"])
        self.assertIsNone(namespace["algorithm_active_key"]("lmsp",selected,state))

    def test_dataset_button_refreshes_on_mode_change_and_uses_concrete_labels(self):
        self.assertIn("algorithm_button_label",call_names(self.methods["_algo_label"]))
        self.assertIn("_refresh_algo_buttons",call_names(self.methods["_toggle_mode"]))
        self.assertIn("_algo_label",call_names(self.methods["_refresh_algo_buttons"]))
        self.assertIn("_algo_bg",call_names(self.methods["_refresh_algo_buttons"]))


if __name__ == "__main__":
    unittest.main()
