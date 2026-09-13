import { app } from "../../scripts/app.js";
import { syncDynamicInputs } from "./dynamic_inputs.js";

function scheduleRefresh(node) {
    if (node._sglangRefreshPending) return;
    node._sglangRefreshPending = true;
    // Configuration and connection callbacks can run before graph links are
    // fully restored. Reconcile once the synchronous graph operation finishes.
    setTimeout(() => {
        try {
            syncDynamicInputs(node);
            refreshPrompts(node);
        } finally {
            node._sglangRefreshPending = false;
        }
    }, 0);
}

export function slotOf(value) {
    return String(value ?? "").split("::", 1)[0].trim();
}

function sourceFor(node, input) {
    let link = node.graph?.links?.[input.link];
    let source = link && node.graph.getNodeById(link.origin_id);
    const visited = new Set();
    while (source?.type === "Reroute" && !visited.has(source.id)) {
        visited.add(source.id);
        link = node.graph.links?.[source.inputs?.[0]?.link];
        source = link && node.graph.getNodeById(link.origin_id);
    }
    return source;
}

export function refreshPrompts(node) {
    let widget = node.widgets?.find(w => w.name === "active_prompt");
    if (!widget) return;
    if (!widget._sglangCombo) {
        // Replace the STRING widget itself: changing its type alone leaves its
        // text-editor mouse handler attached in the current ComfyUI frontend.
        const index = node.widgets.indexOf(widget);
        const replacement = node.addWidget("combo", "active_prompt", widget.value,
            () => {}, { values: [widget.value] });
        node.widgets.splice(node.widgets.indexOf(replacement), 1);
        node.widgets[index] = replacement;
        replacement._sglangCombo = true;
        widget = replacement;
    }
    const selected = slotOf(widget.value);
    const options = [];
    for (const input of node.inputs ?? []) {
        if (!/^system_prompt_\d+$/.test(input.name) || input.link == null) continue;
        const source = sourceFor(node, input);
        const title = source?.title || source?.type || "Connected text";
        options.push(`${input.name} :: ${title}`);
    }
    // Keep a disconnected choice visible; never silently activate another prompt.
    let current = options.find(value => slotOf(value) === selected);
    if (!current) {
        current = `${selected || "system_prompt_1"} :: [disconnected]`;
        options.unshift(current);
    }
    widget.type = "combo";
    widget.options ??= {};
    widget.options.values = options;
    widget.value = current;
    widget.label = "Active system prompt";
    widget.serializeValue = () => slotOf(widget.value);
}

app.registerExtension({
    name: "XavierLAB.SGLangPromptComposer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SGLangPromptComposer") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            refreshPrompts(this);
            scheduleRefresh(this);
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = configured?.apply(this, args);
            scheduleRefresh(this);
            return result;
        };
        const connections = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (...args) {
            const result = connections?.apply(this, args);
            scheduleRefresh(this);
            return result;
        };
        const drawn = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (...args) {
            refreshPrompts(this);
            return drawn?.apply(this, args);
        };
    },
});
