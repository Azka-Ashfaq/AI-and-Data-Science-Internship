import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13, 7.5))
ax.set_xlim(0, 13)
ax.set_ylim(-0.6, 7.5)
ax.axis("off")

NODE = dict(boxstyle="round,pad=0.35", linewidth=1.4)
COLORS = {
    "entry": "#DCE9F9", "process": "#E8F5E9", "tool": "#FFF3CD",
    "gate": "#FADBD8", "end": "#D6DBDF",
}

def box(x, y, w, h, text, kind, ax):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                        linewidth=1.3, edgecolor="#444", facecolor=COLORS[kind])
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)
    return (x, y, w, h)

def arrow(a, b, ax, label=None, style="-|>", color="#333"):
    ax1 = a[0] + a[2] / 2, a[1]
    ax2 = b[0] + b[2] / 2, b[1] + b[3]
    fa = FancyArrowPatch(ax1, ax2, arrowstyle=style, mutation_scale=12, color=color, linewidth=1.2)
    ax.add_patch(fa)
    if label:
        mx, my = (ax1[0] + ax2[0]) / 2, (ax1[1] + ax2[1]) / 2
        ax.text(mx + 0.15, my, label, fontsize=7.5, color=color)

# Nodes
n_client   = box(5.1, 6.7, 2.8, 0.6, "Client ticket (text)", "entry", ax)
n_validate = box(5.1, 5.9, 2.8, 0.6, "validate_input\n(bad-input handling)", "process", ax)
n_classify = box(5.1, 5.1, 2.8, 0.6, "classify_ticket\n(category + priority)", "process", ax)
n_reject   = box(9.6, 5.1, 2.6, 0.6, "reject_early\n(refused / invalid)", "gate", ax)
n_kb       = box(2.0, 5.1, 2.4, 0.6, "retrieve_kb tool\n(kb.json data source,\nDay 2 pattern)", "tool", ax)
n_gen      = box(5.1, 4.15, 2.8, 0.6, "generate_draft\n(Day 3 pattern)", "process", ax)
n_crit     = box(5.1, 3.2, 2.8, 0.6, "critique_draft\nscore 0-100", "process", ax)
n_revise   = box(1.9, 3.2, 2.4, 0.6, "revise\n(retry, max 2)", "process", ax)
n_gate     = box(5.1, 2.1, 2.8, 0.75, "human_checkpoint\nrefund/cancel/billing/\nP1 -> interrupt_before\n(Day 3 pattern)", "gate", ax)
n_send     = box(5.1, 0.95, 2.8, 0.6, "finalize -> send_response\ntool (timeout handled)", "tool", ax)
n_state    = box(9.6, 2.9, 2.9, 1.9, "TicketState\n(checkpointed by\nInMemorySaver,\nDay 3 pattern)\nid, text, category,\npriority, kb_result,\ndraft, score, retries,\nstatus, log[]", "end", ax)
n_end      = box(5.1, -0.15, 2.8, 0.6, "sent / rejected /\nfailed_will_retry", "end", ax)

arrow(n_client, n_validate, ax)
arrow(n_validate, n_classify, ax, "valid")
arrow(n_validate, n_reject, ax, "invalid", style="->", color="#B03A2E")
arrow(n_classify, n_reject, ax, "refusal", style="->", color="#B03A2E")
arrow(n_classify, n_kb, ax)
arrow(n_kb, n_gen, ax)
arrow(n_gen, n_crit, ax)
arrow(n_crit, n_revise, ax, "score<80", style="->", color="#B7950B")
arrow(n_revise, n_gen, ax)
arrow(n_crit, n_gate, ax, "score>=80,\nneeds approval")
ax.annotate("", xy=(6.6, 3.25), xytext=(6.9, 1.55),
            arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1,
                             connectionstyle="arc3,rad=0.5"))
ax.text(7.7, 2.3, "score>=80,\nno approval\nneeded", fontsize=6.5, color="#333", ha="left")
arrow(n_gate, n_send, ax, "approved")
arrow(n_send, n_end, ax)
arrow(n_reject, n_end, ax)

ax.text(1.0, 0.6, "Legend:", fontsize=9, weight="bold")
legend_items = [("entry", "entry point"), ("process", "agent node (mock-LLM reasoning)"),
                ("tool", "external tool / data source"), ("gate", "checkpoint / rejection"),
                ("end", "state / terminal")]
for i, (k, label) in enumerate(legend_items):
    yy = 0.35 - i * 0.32
    ax.add_patch(mpatches.Rectangle((1.0, yy - 0.08), 0.25, 0.2, facecolor=COLORS[k], edgecolor="#444"))
    ax.text(1.35, yy, label, fontsize=7.5, va="center")

ax.set_title("Web3Geeks Ticket-Triage Agent — LangGraph Architecture", fontsize=13, weight="bold", pad=12)
plt.tight_layout()
plt.savefig("architecture_diagram.png", dpi=180)
print("saved")
