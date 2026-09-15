"""Non-LLM preflight: imports, pinned data/tokenizer loading and synthetic resume tests."""
import hashlib,json,pathlib,sys,tempfile,subprocess,importlib.metadata as metadata
ROOT=pathlib.Path(__file__).resolve().parents[3]
P=ROOT/'paper2/phase3'
sys.path.insert(0,str(ROOT/'research'))
import generate_fresh_panel as g
import torch
from datasets import load_dataset
from transformers import AutoTokenizer,AutoConfig,GenerationConfig
from run_phase1c import extract_answer
from math_scoring import math_equal,vote_key

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    assert digest(P/'FROZEN_POLICY.md')=='d6a2835c17d943bc6f07bc50f745c76e175b74287666f806b13d0945fe5e290f'
    assert digest(ROOT/'research/math_scoring.py')=='fd787f23e94d3a01d2a75f48020635093a52c600784400036fb8f53ad6980111'
    assert digest(ROOT/'research/run_phase1c.py')=='132648485b8dfd08c9f3f0d751fe071db3680178cc39849ba321d474c1d362dc'
    expected={'torch':'2.11.0+cu128','transformers':'5.16.1','accelerate':'1.14.0','datasets':'5.0.1','math-verify':'0.8.0','latex2sympy2-extended':'1.10.2','antlr4-python3-runtime':'4.13.2','sympy':'1.14.0','tokenizers':'0.23.1','numpy':'2.5.2'}
    for name,version in expected.items(): assert metadata.version(name)==version,(name,metadata.version(name))
    import test_math_scoring
    test_math_scoring.main()
    assert extract_answer('Final Answer: 1\nFinal Answer: 2')[0]=='2'
    assert extract_answer(r'\boxed{\frac{1}{2}}')[0]==r'\frac{1}{2}'
    assert extract_answer('no answer')[2]=='INVALID_EXTRACTION'
    assert extract_answer('\\boxed{2}\nFinal Answer:')[2]=='INVALID_EXTRACTION'
    assert not math_equal('2','3')
    assert vote_key(None) is None
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 4090'
    assert (torch.ones(2,device='cuda')+1).cpu().tolist()==[2,2]
    dataset=load_dataset(g.DATASET_ID,revision=g.DATASET_REVISION,split='test',cache_dir=str(P/'cache/datasets'))
    assert len(dataset)==500
    historical={}
    seeds=set()
    with (ROOT/'results/raw/generations.jsonl').open() as f:
        for line in f:
            r=json.loads(line); q=r['question_index']; s=r['sample_index']
            assert r['seed']==20260902+q*16+s-1
            seeds.add(r['seed']); historical[q]=r
    assert len(seeds)==8000
    template=(ROOT/'research/pilot_prompt_template_v2.txt').read_text()
    for q,row in enumerate(dataset):
        old=historical[q]
        assert row['problem']==old['question'] and row['answer']==old['reference_answer']
        assert template.format(problem=row['problem'])==old['prompt']
    tokenizer=AutoTokenizer.from_pretrained(g.MODEL_ID,revision=g.MODEL_REVISION,cache_dir=str(P/'cache/model'))
    config=AutoConfig.from_pretrained(g.MODEL_ID,revision=g.MODEL_REVISION,cache_dir=str(P/'cache/model'))
    generation=GenerationConfig.from_pretrained(g.MODEL_ID,revision=g.MODEL_REVISION,cache_dir=str(P/'cache/model'))
    assert config._commit_hash==g.MODEL_REVISION
    assert tokenizer(template.format(problem=dataset[0]['problem']))['input_ids']
    record=dict(historical[0]); record.update(seed=g.derived_seed(0,record['sample_index']),generation_seed=g.derived_seed(0,record['sample_index']),model_revision=g.MODEL_REVISION,dataset_revision=g.DATASET_REVISION)
    # Historical record copied ONLY to a temporary test file. Never prospective output.
    with tempfile.TemporaryDirectory(dir=P) as tmp:
        original=g.RAW_PATH; g.RAW_PATH=pathlib.Path(tmp)/'synthetic_resume.jsonl'
        try:
            assert g.load_completed()==set()
            with g.RAW_PATH.open('a') as output: g.append_record(output,record)
            assert g.load_completed()=={(0,record['sample_index'])}
            for content in [json.dumps(record)+'\n'+json.dumps(record)+'\n','{broken\n',json.dumps(dict(record,seed=0))+'\n',json.dumps(record)]:
                g.RAW_PATH.write_text(content)
                try: g.load_completed()
                except (RuntimeError,KeyError): pass
                else: raise AssertionError('Corrupt resume accepted')
        finally: g.RAW_PATH=original
    # Independent per-row RNGs: verify alone versus batched, with synthetic probabilities only.
    probabilities=torch.tensor([[0.1,0.2,0.7],[0.1,0.2,0.7]],device='cuda')
    with g.independent_multinomial_streams([30260915,30260916]):
        batch=torch.stack([torch.multinomial(probabilities,1) for _ in range(10)])
    with g.independent_multinomial_streams([30260916]):
        alone=torch.stack([torch.multinomial(probabilities[1:],1) for _ in range(10)])
    assert torch.equal(batch[:,1],alone[:,0])
    assert not g.RAW_PATH.exists(), 'Prospective output unexpectedly exists'
    payload={'status':'PASS','prospective_generations':0,'python':sys.version,'executable':sys.executable,'torch_cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0),'gpu_vram_bytes':torch.cuda.get_device_properties(0).total_memory,'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),'versions':{d.metadata['Name']:d.version for d in metadata.distributions()},'dataset_questions':len(dataset),'dataset_order_matches_paper1':True,'model_revision':config._commit_hash,'dataset_revision':g.DATASET_REVISION,'generation_defaults':generation.to_dict(),'checks':['scoring fixtures','extraction priority and invalid marker','CUDA tensor operation (no LLM)','all dataset questions, references and rendered prompts match Paper 1','pinned tokenizer/config load','valid resume','duplicate rejection','malformed record rejection','seed mismatch rejection','no prospective raw file']}
    (P/'runtime_audit.json').write_text(json.dumps(payload,indent=2,default=str)+'\n')
    print('NON-LLM PREFLIGHT PASS')
if __name__=='__main__': main()
