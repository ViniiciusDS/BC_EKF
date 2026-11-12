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

    def test_button_text_fits_even_with_larger_system_font(self):
        source = module_ast("src/ui/botton.py")
        func = public_function(source, "fit_button_label_size")
        code = compile(ast.Module(body=[func], type_ignores=[]), "<fit_button_label_size>", "exec")
        scope = {}
        exec(code, scope)
        fit = scope["fit_button_label_size"]
        self.assertEqual(fit(150, 22, 220, 44), (150, 22))
        width, height = fit(260, 30, 220, 44)
        self.assertLessEqual(width, 196)  # 12 px de cada lado
        self.assertLessEqual(height, 32)
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)
        self.assertAlmostEqual(width / height, 260/30, delta=.2)
        with self.assertRaises(ValueError):
            fit(0, 10, 220, 44)
        button_cls = next(n for n in source.body
                          if isinstance(n, ast.ClassDef) and n.name == "Button")
        draw = next(n for n in button_cls.body
                    if isinstance(n, ast.FunctionDef) and n.name == "draw")
        self.assertIn("fit_button_label_size", call_names(draw))
        self.assertIn("smoothscale", call_names(draw))
        self.assertIn("blit", call_names(draw))


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
