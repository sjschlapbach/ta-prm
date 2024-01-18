import pytest
from .ta_prm_demo import ta_prm_demo
from .ta_prm_random import ta_prm_random
from .ta_prm_worst_case import ta_prm_worst_case


class TestTAPRMDemo:
    def test_ta_prm_demo(self):
        ta_prm_demo()

    def test_ta_prm_random(self):
        ta_prm_random()

    def test_ta_prm_worst_case(self):
        ta_prm_worst_case()
