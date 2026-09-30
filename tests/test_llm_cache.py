"""模型响应缓存 —— 让「实测」与「可复现」同时成立。

⚠️ 这套机制的存在理由：手写一份「模型画像」（比如「硬条款 95%、软条款 85%」）
   得到的矩阵量的是那份画像，不是模型。真跑一次、逐条落盘、之后重放，
   才是既省钱又不自说自话的做法。
"""

import json

import pytest

from csagent.evaluation.llm_cache import CachedLlm, RecordingLlm
from csagent.llm import StubLlm


def test_recording_llm_records_every_prompt(tmp_path):
    path = tmp_path / "cache.json"
    rec = RecordingLlm(StubLlm(replies={"甲": {"v": 1}, "乙": {"v": 2}}), path)
    rec.complete_json("S", "甲的情况", dict)
    rec.complete_json("S", "乙的情况", dict)
    assert json.loads(path.read_text(encoding="utf-8")) == {"甲的情况": {"v": 1},
                                                            "乙的情况": {"v": 2}}


def test_recording_llm_does_not_call_twice_for_the_same_prompt(tmp_path):
    rec = RecordingLlm(StubLlm(replies={"甲": {"v": 1}}), tmp_path / "cache.json")
    rec.complete_json("S", "甲", dict)
    rec.complete_json("S", "甲", dict)
    assert rec.calls == 1 and rec.reused == 1


def test_recording_llm_persists_after_each_call(tmp_path):
    """每调一次就落盘 —— 一次完整跑要几分钟，中途断掉不该丢掉已花的钱。"""
    path = tmp_path / "cache.json"
    RecordingLlm(StubLlm(replies={"甲": {"v": 1}}), path).complete_json("S", "甲", dict)
    assert path.exists() and "甲" in json.loads(path.read_text(encoding="utf-8"))


def test_cached_llm_replays_the_recorded_reply(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"甲": {"v": 1}}, ensure_ascii=False), encoding="utf-8")
    assert CachedLlm(path).complete_json("S", "甲", dict) == {"v": 1}


def test_cached_llm_fails_loudly_on_a_missing_prompt(tmp_path):
    """缓存与评测集不同步时**大声失败**。

    返回一个默认值会让评测在「一半样本没测到」的情况下跑完，
    还显得一切正常 —— 那正是最难发现的一类错。
    """
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"甲": {"v": 1}}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(KeyError):
        CachedLlm(path).complete_json("S", "乙", dict)


def test_cached_llm_fails_loudly_when_there_is_no_cache_at_all(tmp_path):
    with pytest.raises(FileNotFoundError):
        CachedLlm(tmp_path / "nope.json")


def test_round_trip_survives_json_encoding(tmp_path):
    """中文提示词过一遍 JSON 必须原样回来 —— 否则缓存永远命中不了。"""
    path = tmp_path / "cache.json"
    rec = RecordingLlm(StubLlm(replies={"这课我死活上不了": {"risk": "high"}}), path)
    rec.complete_json("系统", "这课我死活上不了，钱白花了", dict)
    assert CachedLlm(path).complete_json(
        "系统", "这课我死活上不了，钱白花了", dict) == {"risk": "high"}
