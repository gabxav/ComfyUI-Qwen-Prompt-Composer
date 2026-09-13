import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
const source = await readFile(new URL('../custom_node/web/dynamic_inputs.js', import.meta.url), 'utf8');
const { syncDynamicInputs } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
function node() {
    return {
        inputs: [...Array.from({length:16},(_,i)=>({name:`system_prompt_${i+1}`,type:'STRING',link:null})),
            ...Array.from({length:8},(_,i)=>({name:`image_${i+1}`,type:'IMAGE',link:null})),{name:'video',type:'IMAGE',link:99}],
        graph:{links:{99:{target_slot:24}}},
        addInput(name,type){this.inputs.push({name,type,link:null});},
        removeInput(i){assert.equal(this.inputs[i].link,null);this.inputs.splice(i,1);},
        setSize(){},computeSize(){return [400,400];},setDirtyCanvas(){},
    };
}
function connect(n,name,id){const index=n.inputs.findIndex(i=>i.name===name);n.inputs[index].link=id;n.graph.links[id]={target_slot:index};}
function validLinks(n){n.inputs.forEach((i,index)=>{if(i.link!=null)assert.equal(n.graph.links[i.link].target_slot,index);});}
test('one spare socket per family, preserving video destination',()=>{
 const n=node();syncDynamicInputs(n);assert.deepEqual(n.inputs.map(i=>i.name),['system_prompt_1','image_1','video']);validLinks(n);
 connect(n,'system_prompt_1',1);syncDynamicInputs(n);assert.equal(n.inputs[1].name,'system_prompt_2');validLinks(n);
 connect(n,'image_1',2);syncDynamicInputs(n);assert(n.inputs.some(i=>i.name==='image_2'));validLinks(n);
});
test('sparse saved workflows and disconnects retain socket identity',()=>{
 const n=node();connect(n,'system_prompt_7',7);connect(n,'image_3',3);syncDynamicInputs(n);
 assert.equal(n.inputs.filter(i=>i.name.startsWith('system_prompt_')).length,8);validLinks(n);
 n.inputs.find(i=>i.name==='system_prompt_7').link=null;delete n.graph.links[7];syncDynamicInputs(n);
 assert.equal(n.inputs.filter(i=>i.name.startsWith('system_prompt_')).length,1);assert.equal(n.inputs.find(i=>i.name==='image_3').link,3);validLinks(n);
});
test('full capacity does not add backend-unsupported inputs; reconciliation is idempotent',()=>{
 const n=node();connect(n,'system_prompt_16',16);connect(n,'image_8',8);syncDynamicInputs(n);
 const saved=JSON.stringify(n.inputs);syncDynamicInputs(n);assert.equal(JSON.stringify(n.inputs),saved);assert.equal(n.inputs.length,25);validLinks(n);
});
