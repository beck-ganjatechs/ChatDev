from __future__ import annotations

import json
from typing import Dict, Any, List

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from chatdev.chat_env import ChatEnv
from chatdev.utils import log_visualize


class ChainState(BaseModel):
    env: ChatEnv
    idx: int = 0
    chain: List[Dict[str, Any]] = Field(default_factory=list)
    config_phase: Dict[str, Any] = Field(default_factory=dict)
    role_prompts: Dict[str, str] = Field(default_factory=dict)
    model_type: Any = None
    log_filepath: str = ""


def _apply_phase(state: ChainState) -> ChainState:
    phase_item = state.chain[state.idx]
    phase = phase_item["phase"]
    phase_type = phase_item["phaseType"]

    # Defer import to keep original modules untouched
    from chatdev.phase import __dict__ as simple_phases
    from chatdev.composed_phase import __dict__ as composed_phases

    if phase_type == "SimplePhase":
        phase_cls = simple_phases.get(phase)
        if phase_cls is None:
            raise RuntimeError(f"Phase '{phase}' not implemented")
        conf = state.config_phase[phase]
        assistant_role_name = conf['assistant_role_name']
        user_role_name = conf['user_role_name']
        phase_prompt = "\n\n".join(conf['phase_prompt'])
        instance = phase_cls(
            assistant_role_name=assistant_role_name,
            user_role_name=user_role_name,
            phase_prompt=phase_prompt,
            role_prompts=state.role_prompts,
            phase_name=phase,
            model_type=state.model_type,
            log_filepath=state.log_filepath
        )
        # preserve existing semantics
        max_turn_step = int(phase_item.get("max_turn_step", 10))
        need_reflect = str(phase_item.get("need_reflect", "False")).lower() == "true"
        new_env = instance.execute(state.env, max_turn_step, need_reflect)
        state.env = new_env
    elif phase_type == "ComposedPhase":
        compose_cls = composed_phases.get(phase)
        if compose_cls is None:
            raise RuntimeError(f"Composed phase '{phase}' not implemented")
        instance = compose_cls(
            phase_name=phase,
            cycle_num=phase_item.get("cycleNum", 1),
            composition=phase_item.get("Composition", []),
            config_phase=state.config_phase,
            config_role={k: v for k, v in state.role_prompts.items()},
            model_type=state.model_type,
            log_filepath=state.log_filepath
        )
        new_env = instance.execute(state.env)
        state.env = new_env
    else:
        raise RuntimeError(f"Unsupported phaseType: {phase_type}")

    return state


def _inc(state: ChainState) -> ChainState:
    state.idx += 1
    return state


def build_graph() -> StateGraph:
    graph = StateGraph(ChainState)
    graph.add_node("phase", _apply_phase)
    graph.add_node("inc", _inc)

    def should_continue(s: ChainState):
        return None if s.idx >= len(s.chain) else "phase"

    graph.set_entry_point("phase")
    graph.add_conditional_edges(
        "inc",
        should_continue,
        {None: END, "phase": "phase"}
    )
    graph.add_edge("phase", "inc")

    return graph


def run_langgraph_engine(config_path: str,
                         config_phase_path: str,
                         config_role_path: str,
                         task_prompt: str,
                         project_name: str,
                         org_name: str,
                         model_type,
                         code_path: str | None,
                         log_filepath: str,
                         existing_env: ChatEnv) -> ChatEnv:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(config_phase_path, "r", encoding="utf-8") as f:
        config_phase = json.load(f)
    with open(config_role_path, "r", encoding="utf-8") as f:
        config_role = json.load(f)

    role_prompts: Dict[str, str] = {role: "\n".join(prompts) for role, prompts in config_role.items()}

    # reuse env prepared in preprocessing
    env = existing_env
    env.env_dict["task_prompt"] = task_prompt
    env.env_dict["log_filepath"] = log_filepath

    initial = ChainState(
        env=env,
        idx=0,
        chain=config.get("chain", []),
        config_phase=config_phase,
        role_prompts=role_prompts,
        model_type=model_type,
        log_filepath=log_filepath,
    )

    memory = MemorySaver()
    app = build_graph().compile(checkpointer=memory)

    for event in app.stream(initial, stream_mode="values"):
        if isinstance(event, ChainState):
            log_visualize(f"[LangGraph] phase idx={event.idx}")
    final_state = app.get_state({}).values
    return final_state.env