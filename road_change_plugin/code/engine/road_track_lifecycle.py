"""Event-constrained lifecycle. Auto is an observation, never a state transition."""
import math


def resolve_lifecycle(periods, events):
    order={str(p):i for i,p in enumerate(periods)}
    if not events:return {},[]
    for event in events:
        if event["before"] not in order or event["after"] not in order:
            raise ValueError("变化事件引用了未知期次")
        if order[event["after"]]!=order[event["before"]]+1:
            raise ValueError("生命周期仅接受相邻期变化事件")
    grouped={}
    for event in events:grouped.setdefault(event["after"],[]).append(event)
    selected={}
    audit=[]
    for period,values in grouped.items():
        values=sorted(values,key=lambda e:(e.get("origin")=="AUTO",str(e["source"])))
        selected[period]=values[0]
        for other in values[1:]:
            audit.append(dict(period=period,source=other["source"],qa_reason="duplicate_event_constraint",
                              selected_source=values[0]["source"]))
    ordered=sorted(selected.values(),key=lambda e:order[e["after"]])
    first=ordered[0]
    present=first["kind"]!="added"
    width=float(first["width_before"]) if present else 0.
    source=first["source"];origin=first.get("origin","event");profile=first.get("profile")
    states={};last_event=None
    for period in map(str,periods):
        event=selected.get(period)
        state_source="inherited_event" if states else "initial_event_constraint"
        conflict=""
        if event:
            kind=event["kind"]
            before=states[event["before"]]
            # Only directly adjacent positive width constraints share a boundary.
            # No missing observation or later nonadjacent event rewrites history.
            if (last_event and last_event["after"]==event["before"] and before["present"]
                    and kind in ("widened","narrowed","removed") and float(event["width_before"])>0):
                shared=(before["width"]+float(event["width_before"]))/2
                before["width"]=shared;width=shared
            if kind=="added" and present or kind=="removed" and not present:
                conflict="adjacent_state_conflict_resolved"
            elif kind in ("widened","narrowed") and not present:
                conflict="width_event_on_absent_track"
            else:
                present=kind!="removed"
                width=float(event["width_after"]) if present else 0.
                source=event["source"];origin=event.get("origin","event");profile=event.get("profile")
                state_source="explicit_event"
                last_event=event
            if conflict:
                audit.append(dict(period=period,source=event["source"],qa_reason=conflict,
                                  selected_present=present,selected_source=source))
                state_source="inherited_event_conflict"
        if present and (not math.isfinite(width) or width<=0):
            raise ValueError("道路存在状态缺少有效宽度")
        states[period]=dict(present=present,width=width,source=source,state_source=state_source,
                            event_origin=origin,gt_width_profile=profile,event_conflict=conflict)
    return states,audit
