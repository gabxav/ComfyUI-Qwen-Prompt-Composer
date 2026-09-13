// Keep stable socket names and one spare input after the last connection.
export function syncDynamicInputs(node) {
    const before = node.inputs.length;
    for (const [prefix, type, limit] of [["system_prompt_", "STRING", 16], ["image_", "IMAGE", 8]]) {
        const belongs = input => input.name.startsWith(prefix) && /^\d+$/.test(input.name.slice(prefix.length));
        const connected = node.inputs.filter(input => belongs(input) && input.link != null);
        const last = Math.max(0, ...connected.map(input => Number(input.name.slice(prefix.length))));
        const keep = Math.min(limit, last + 1);
        // Remove only unused trailing inputs. Never rename or disconnect a socket.
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const input = node.inputs[i];
            if (belongs(input) && input.link == null && Number(input.name.slice(prefix.length)) > keep) {
                node.removeInput(i);
            }
        }
        for (let number = 1; number <= keep; number++) {
            const name = `${prefix}${number}`;
            if (node.inputs.some(input => input.name === name)) continue;
            const previous = node.inputs.findIndex(input => input.name === `${prefix}${number - 1}`);
            node.addInput(name, type);
            if (previous >= 0) {
                const added = node.inputs.pop();
                node.inputs.splice(previous + 1, 0, added);
            }
        }
    }
    // Inserting a system socket can move image/video sockets. Repair their link
    // indices so UI save/load and graphToPrompt keep the same destinations.
    node.inputs.forEach((input, index) => {
        const link = node.graph?.links?.[input.link];
        if (link) link.target_slot = index;
    });
    if (node.inputs.length !== before) {
        const size = node.computeSize();
        size[0] = Math.max(node.size?.[0] ?? 0, size[0]);
        node.setSize(size);
        node.setDirtyCanvas(true, true);
    }
}
