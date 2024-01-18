import pytest
from .ta_prm_demo import ta_prm_demo
from .ta_prm_random import ta_prm_random


class TestTAPRMDemo:
    def test_ta_prm_demo(self):
        ta_prm_demo()

    def test_ta_prm_random(self):
        ta_prm_random()
