"""Skill 加载器：扫描 src/skills/*/SKILL.md，front-matter 写元数据，
正文是按需加载的过程知识（渐进式披露，M2）。
"""
from __future__ import annotations

from pathlib import Path

import yaml

SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def _parse_front_matter(text: str):
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    meta = yaml.safe_load(text[3:end]) or {}
    body = text[end + 4:].lstrip("\n")
    return meta, body


def list_skills(skills_dir=SKILLS_DIR):
    out = []
    for md in sorted(Path(skills_dir).glob("*/SKILL.md")):
        meta, _ = _parse_front_matter(md.read_text(encoding="utf-8"))
        meta["name"] = meta.get("name", md.parent.name)
        meta["path"] = str(md)
        out.append(meta)
    return out


def load_skill(name: str, skills_dir=SKILLS_DIR):
    md = Path(skills_dir) / name / "SKILL.md"
    meta, body = _parse_front_matter(md.read_text(encoding="utf-8"))
    meta["name"] = meta.get("name", name)
    return {"meta": meta, "body": body}


def match_skills(task_text: str, skills_dir=SKILLS_DIR):
    """description / when_to_use 命中关键词时返回匹配的 skill 正文。"""
    text = task_text.lower()
    hits = []
    for meta in list_skills(skills_dir):
        blob = " ".join([str(meta.get("description", "")),
                         str(meta.get("when_to_use", "")),
                         str(meta.get("name", ""))]).lower()
        if any(kw.strip() and kw.strip().lower() in text
               for kw in blob.replace("，", ",").split(",")):
            hits.append(load_skill(meta["name"], skills_dir)["body"])
    return hits


class SkillLoader:
    """对模块级函数的面向对象封装，方便自检与外部按目录实例化。

    eval/run.py 通过 ``SkillLoader(skills_dir).list_skills()`` 校验元数据。
    """

    def __init__(self, skills_dir=SKILLS_DIR):
        self.skills_dir = Path(skills_dir)

    def list_skills(self):
        return list_skills(self.skills_dir)

    def load_skill(self, name: str):
        return load_skill(name, self.skills_dir)

    def match_skills(self, task_text: str):
        return match_skills(task_text, self.skills_dir)


if __name__ == "__main__":
    for s in list_skills():
        print(s["name"], "-", s.get("description", "")[:60])
