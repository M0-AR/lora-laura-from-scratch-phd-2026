import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import count_params, lora_count, valid_plain, valid_pirate, valid_polite

def test_counts():
    c = count_params()
    assert c["total"] == 1644608, c
    assert lora_count(4) == 18688

def test_validators():
    assert valid_plain(["the", "pilot", "sells", "a", "ship", "."])
    assert not valid_plain(["R", "ye", "pilot", "sells", "a", "ship", "matey", "."])
    assert valid_pirate(["R", "ye", "pilot", "sells", "a", "ship", "matey", "."])
    assert valid_polite(["please", "the", "captain", "kindly", "guards", "the", "net", "thank", "you", "."])
