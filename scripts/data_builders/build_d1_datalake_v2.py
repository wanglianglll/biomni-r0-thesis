
from __future__ import annotations
import ast, json, random, re, hashlib
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

ROOT = Path('/root/autodl-tmp/Biomni-main')
DL = ROOT / 'data/biomni_data/data_lake'
OUT = ROOT / 'data/sft_d1_datalake_v2'
SEED = 42
TRAIN_RATIO = 0.98
random.seed(SEED)
OUT.mkdir(parents=True, exist_ok=True)
SYSTEM = 'You are Biomni, a biomedical assistant. Use the provided structured evidence and answer concisely. End with exactly one FINAL ANSWER line.'

def clean(x):
    if x is None: return ''
    try:
        if pd.isna(x): return ''
    except Exception:
        pass
    return re.sub(r'\s+', ' ', str(x)).strip()

def safe_list(x):
    if isinstance(x, list): return [clean(v) for v in x if clean(v)]
    s=clean(x)
    if not s: return []
    try:
        v=ast.literal_eval(s)
        if isinstance(v, list): return [clean(i) for i in v if clean(i)]
    except Exception:
        pass
    if '|' in s: return [clean(i) for i in s.split('|') if clean(i)]
    if ',' in s: return [clean(i) for i in s.split(',') if clean(i)]
    return [s]

def valid_gene(g): return bool(re.fullmatch(r'[A-Za-z][A-Za-z0-9._-]{1,30}', clean(g)))
def valid_ensg(g): return bool(re.fullmatch(r'ENSG\d{11}', clean(g)))
def valid_rsid(x): return bool(re.fullmatch(r'rs\d+', clean(x), flags=re.I))

def make(task_type, source_file, source_row_id, user, answer, evidence=None, metadata=None, quality_flags=None):
    answer=clean(answer)
    if not user or not answer: return None
    assistant = ''
    if evidence:
        assistant += f'Key evidence: {clean(evidence)}\n'
    assistant += f'FINAL ANSWER: {answer}'
    return {
        'dataset': 'D1_datalake_v2',
        'task_type': task_type,
        'source': source_file.rsplit('.',1)[0],
        'source_file': source_file,
        'source_row_id': source_row_id,
        'answer': answer,
        'metadata': metadata or {},
        'quality_flags': quality_flags or [],
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': user},
            {'role': 'assistant', 'content': assistant},
        ],
    }

def add_sample(samples, sample):
    if sample: samples.append(sample)

def sample_df(df, n, seed=SEED):
    if len(df) <= n: return df.copy()
    return df.sample(n, random_state=seed).copy()

def build_gwas(limit=18000):
    df=pd.read_pickle(DL/'gwas_catalog.pkl')
    cols=['DISEASE/TRAIT','SNPS','MAPPED_GENE','CHR_ID','CHR_POS','CONTEXT','P-VALUE','REPORTED GENE(S)']
    df=df[cols].dropna(subset=['DISEASE/TRAIT','SNPS','MAPPED_GENE']).copy()
    for c in cols: df[c]=df[c].map(clean)
    df=df[df['SNPS'].map(valid_rsid) & df['MAPPED_GENE'].map(valid_gene) & (df['DISEASE/TRAIT'].str.len()>3)]
    df=df.drop_duplicates(subset=['DISEASE/TRAIT','SNPS','MAPPED_GENE'])
    df=sample_df(df, limit)
    out=[]
    for idx,row in df.iterrows():
        user=(f'For the GWAS trait "{row["DISEASE/TRAIT"]}", identify the mapped gene for the associated variant.\n'
              f'Evidence:\n- SNP: {row["SNPS"]}\n- Chromosome: {row["CHR_ID"]}\n- Position: {row["CHR_POS"]}\n- Variant context: {row["CONTEXT"]}\n- Reported gene(s): {row["REPORTED GENE(S)"] or "N/A"}\n- P-value: {row["P-VALUE"]}\n\nReturn the mapped gene.')
        ev=f'{row["SNPS"]} is mapped to {row["MAPPED_GENE"]} for {row["DISEASE/TRAIT"]} in the GWAS catalog.'
        add_sample(out, make('gwas_trait_to_gene','gwas_catalog.pkl',int(idx),user,row['MAPPED_GENE'],ev,{'trait':row['DISEASE/TRAIT'],'snp':row['SNPS']}))
    return out

def build_variant(limit=8000):
    df=pd.read_parquet(DL/'variant_table.parquet')
    for c in df.columns: df[c]=df[c].map(clean)
    df=df[df['RS'].map(valid_rsid)].drop_duplicates(subset=['RS'])
    df=sample_df(df,limit)
    out=[]
    for idx,row in df.iterrows():
        user=(f'Summarize the variant record below.\nEvidence:\n- RS ID: {row.RS}\n- Internal ID: {row.ID}\n- Chromosome: {row.CHR}\n- Position: {row.POS}\n- Alleles: {row.A1}/{row.A2}\n- MAF: {row.MAF}\n\nReturn the RS ID.')
        ev=f'The variant table lists {row.RS} at chr{row.CHR}:{row.POS} with alleles {row.A1}/{row.A2}.'
        add_sample(out, make('variant_record_to_rsid','variant_table.parquet',int(idx),user,row.RS,ev,{'chr':row.CHR,'pos':row.POS}))
    return out

def build_gene_info(limit=15000):
    df=pd.read_parquet(DL/'gene_info.parquet')
    df=df.dropna(subset=['gene_id','gene_name','chr','gene_start','gene_end']).copy()
    for c in ['gene_id','gene_name','chr','gene_start','gene_end','strand','gene_type','transcript_is_canonical']: df[c]=df[c].map(clean)
    df=df[df['gene_name'].map(valid_gene)].sort_values(['gene_id','transcript_is_canonical'], ascending=[True,False]).drop_duplicates('gene_id')
    df=sample_df(df,limit)
    out=[]
    for idx,row in df.iterrows():
        strand='+' if row.strand=='1' else '-' if row.strand=='-1' else row.strand
        user=(f'Given this Ensembl gene record, return the gene symbol.\nEvidence:\n- Ensembl gene ID: {row.gene_id}\n- Gene symbol: {row.gene_name}\n- Chromosome: {row.chr}\n- Start: {row.gene_start}\n- End: {row.gene_end}\n- Strand: {strand}\n- Gene type: {row.gene_type}')
        ev=f'{row.gene_id} is annotated as {row.gene_name}, a {row.gene_type} gene on chromosome {row.chr}.'
        add_sample(out, make('gene_record_to_symbol','gene_info.parquet',int(idx),user,row.gene_name,ev,{'gene_id':row.gene_id}))
    return out

def build_disgenet(limit_disease=8000, limit_gene=8000):
    df=pd.read_parquet(DL/'DisGeNET.parquet').dropna().copy()
    rows=[]
    for idx,row in df.iterrows():
        disorder=clean(row['Disorder']); genes=[g for g in safe_list(row['Genes']) if valid_gene(g)]
        if disorder and genes: rows.append((idx,disorder,list(dict.fromkeys(genes))))
    random.shuffle(rows)
    out=[]
    for idx,disorder,genes in rows[:limit_disease]:
        ans=', '.join(genes[:8])
        user=f'List candidate genes associated with the disorder "{disorder}" according to the disease-gene knowledge base.'
        ev=f'The DisGeNET entry for {disorder} includes {ans}.'
        add_sample(out, make('disease_to_genes','DisGeNET.parquet',int(idx),user,ans,ev,{'disorder':disorder,'answer_genes':genes[:8]},['multi_gene_answer']))
    gene_pairs=[]
    for idx,disorder,genes in rows:
        for g in genes[:5]: gene_pairs.append((idx,g,disorder,genes[:8]))
    random.shuffle(gene_pairs)
    for idx,g,disorder,genes in gene_pairs[:limit_gene]:
        user=f'Gene {g} appears in a disease-gene knowledge base. Name one associated disorder.'
        ev=f'{g} is listed among candidate genes for {disorder}.'
        add_sample(out, make('gene_to_disorder','DisGeNET.parquet',int(idx),user,disorder,ev,{'gene':g,'disorder':disorder}))
    return out

def build_omim(limit=14000):
    df=pd.read_parquet(DL/'omim.parquet').copy()
    df=df.dropna(subset=['MIM Number']).copy()
    for c in df.columns: df[c]=df[c].map(clean)
    df=df[(df['Approved Gene Symbol']!='') | (df['Gene Name']!='') | (df['Phenotypes']!='')]
    df=df.drop_duplicates(subset=['MIM Number','Approved Gene Symbol','Gene Name','Phenotypes'])
    df=sample_df(df,limit)
    out=[]
    for idx,row in df.iterrows():
        mim=re.sub(r'\.0$','',clean(row['MIM Number']))
        if not re.fullmatch(r'\d{6}',mim): continue
        symbol=row['Approved Gene Symbol'] or row['Gene/Locus And Other Related Symbols'] or row['Gene Name']
        pheno=row['Phenotypes'] or 'not specified'
        user=(f'Provide the OMIM identifier from this OMIM-style record.\nEvidence:\n- Gene/locus: {symbol}\n- Gene name: {row["Gene Name"] or "N/A"}\n- Phenotypes: {pheno}\n- Chromosome: {row["Chromosome"] or "N/A"}\n- Cytogenetic location: {row["Cyto Location"] or "N/A"}')
        ev=f'The record assigns MIM number {mim} to {symbol or row["Gene Name"]}.'
        add_sample(out, make('omim_record_to_mim','omim.parquet',int(idx),user,f'MIM {mim}',ev,{'mim_number':mim,'entity':symbol}))
    return out

def build_proteinatlas(limit=12000):
    df=pd.read_csv(DL/'proteinatlas.tsv',sep='\t',low_memory=False)
    cols=['Gene','Ensembl','Gene description','Protein class','Biological process','Molecular function','Disease involvement','RNA tissue specificity','RNA tissue distribution','Subcellular location']
    df=df[cols].dropna(subset=['Gene']).copy()
    for c in cols: df[c]=df[c].map(clean)
    df=df[df['Gene'].map(valid_gene)].drop_duplicates('Gene')
    df=sample_df(df,limit)
    out=[]
    for idx,row in df.iterrows():
        user=(f'Summarize the Human Protein Atlas profile and return the gene symbol.\nEvidence:\n- Gene: {row.Gene}\n- Ensembl: {row.Ensembl}\n- Description: {row["Gene description"]}\n- Protein class: {row["Protein class"] or "N/A"}\n- Biological process: {row["Biological process"] or "N/A"}\n- Molecular function: {row["Molecular function"] or "N/A"}\n- Disease involvement: {row["Disease involvement"] or "N/A"}\n- RNA tissue specificity: {row["RNA tissue specificity"] or "N/A"}\n- Subcellular location: {row["Subcellular location"] or "N/A"}')
        ev=f'Human Protein Atlas identifies this entry as gene {row.Gene} ({row.Ensembl}).'
        add_sample(out, make('protein_atlas_gene_profile','proteinatlas.tsv',int(idx),user,row.Gene,ev,{'ensembl':row.Ensembl}))
    return out

def build_broad(limit=7000):
    df=pd.read_parquet(DL/'broad_repurposing_hub_phase_moa_target_info.parquet').copy()
    for c in df.columns: df[c]=df[c].map(clean)
    df=df[(df['pert_iname']!='') & (df['target']!='')].drop_duplicates(subset=['pert_iname','target','moa'])
    df=sample_df(df,limit)
    out=[]
    for idx,row in df.iterrows():
        targets=[t for t in safe_list(row.target) if valid_gene(t)]
        if not targets: continue
        ans=', '.join(targets[:8])
        user=(f'Identify molecular targets for the compound below.\nEvidence:\n- Compound: {row.pert_iname}\n- Mechanism of action: {row.moa or "N/A"}\n- Clinical phase: {row.clinical_phase or "N/A"}\n- Disease area: {row.disease_area or "N/A"}\n- Indication: {row.indication or "N/A"}\n\nReturn the listed target gene symbols.')
        ev=f'The Broad Repurposing Hub lists {row.pert_iname} targets as {ans}.'
        add_sample(out, make('compound_to_targets','broad_repurposing_hub_phase_moa_target_info.parquet',int(idx),user,ans,ev,{'compound':row.pert_iname,'targets':targets[:8]},['multi_gene_answer']))
    return out

def build_ddinter(limit_total=12000):
    files=sorted(DL.glob('ddinter_*.csv'))
    per=max(1,limit_total//len(files))
    out=[]
    for fp in files:
        df=pd.read_csv(fp,low_memory=False)
        for c in df.columns: df[c]=df[c].map(clean)
        df=df[(df['Drug_A']!='') & (df['Drug_B']!='') & (df['Level']!='')].drop_duplicates(subset=['Drug_A','Drug_B','Level'])
        df=sample_df(df,per,SEED+len(out))
        for idx,row in df.iterrows():
            user=(f'Identify the reported drug-drug interaction severity.\nEvidence:\n- Drug A: {row.Drug_A}\n- Drug B: {row.Drug_B}\n- DDInter category file: {fp.name}\n\nReturn the interaction level.')
            ev=f'DDInter reports the {row.Drug_A} and {row.Drug_B} interaction level as {row.Level}.'
            add_sample(out, make('drug_interaction_level',fp.name,int(idx),user,row.Level,ev,{'drug_a':row.Drug_A,'drug_b':row.Drug_B}))
    return out[:limit_total]

def build_interactions(limit_total=12000):
    specs=[('affinity_capture-ms.parquet',4000),('co-fractionation.parquet',3000),('proximity_label-ms.parquet',2500),('genetic_interaction.parquet',2500)]
    out=[]
    for fname,lim in specs:
        df=pd.read_parquet(DL/fname)
        for c in df.columns: df[c]=df[c].map(clean)
        df=df[(df['gene_a_id']!='') & (df['gene_b_id']!='')].drop_duplicates(subset=['gene_a_id','gene_b_id','experimental_system_type'])
        df=sample_df(df,lim,SEED+lim)
        for idx,row in df.iterrows():
            user=(f'Summarize this gene interaction record and return the interacting gene pair.\nEvidence:\n- Gene A: {row.gene_a_id}\n- Gene B: {row.gene_b_id}\n- Experimental system type: {row.experimental_system_type}\n- Throughput: {row.throughput_type}\n- PubMed: {row.pubmed_id}\n- Score: {row.experimental_score}')
            ans=f'{row.gene_a_id} -- {row.gene_b_id}'
            ev=f'The {fname} record links {row.gene_a_id} with {row.gene_b_id}.'
            add_sample(out, make('gene_interaction_pair',fname,int(idx),user,ans,ev,{'gene_a':row.gene_a_id,'gene_b':row.gene_b_id},['pair_answer']))
    return out[:limit_total]

def build_genesets(limit_total=12000):
    files=['msigdb_human_c2_curated_geneset.parquet','msigdb_human_c5_ontology_geneset.parquet','msigdb_human_c7_immunologic_signature_geneset.parquet','mousemine_m5_ontology_geneset.parquet']
    per=limit_total//len(files)
    out=[]
    for fname in files:
        df=pd.read_parquet(DL/fname)
        for c in df.columns: df[c]=df[c].map(clean)
        df=df[(df['chromosome_id']!='') & (df['geneSymbols']!='')].drop_duplicates('chromosome_id')
        df=sample_df(df,per,SEED+len(out))
        for idx,row in df.iterrows():
            genes=[g for g in safe_list(row.geneSymbols) if valid_gene(g)]
            if len(genes)<3: continue
            ans=', '.join(genes[:10])
            user=(f'List representative genes from the gene set below.\nEvidence:\n- Gene set: {row.chromosome_id}\n- Collection: {row.collection}\n- Exact source: {row.exactSource or "N/A"}\n- External URL: {row.externalDetailsURL or "N/A"}\n\nReturn up to ten gene symbols from this set.')
            ev=f'The gene set {row.chromosome_id} contains representative genes including {ans}.'
            add_sample(out, make('geneset_to_genes',fname,int(idx),user,ans,ev,{'geneset':row.chromosome_id,'collection':row.collection},['multi_gene_answer']))
    return out[:limit_total]

def build_mirna(limit=8000):
    df=pd.read_parquet(DL/'miRTarBase_microRNA_target_interaction.parquet')
    for c in df.columns: df[c]=df[c].map(clean)
    df=df[(df['miRNA']!='') & (df['Target Gene']!='')].drop_duplicates(subset=['miRNA','Target Gene','Experiments'])
    df=sample_df(df,limit)
    out=[]
    for idx,row in df.iterrows():
        user=(f'Identify the target gene in this miRTarBase interaction.\nEvidence:\n- miRNA: {row.miRNA}\n- Species: {row["Species (miRNA)"]}\n- Experiments: {row.Experiments}\n- Support type: {row["Support Type"]}\n- PMID: {row["References (PMID)"]}\n\nReturn the target gene.')
        ev=f'miRTarBase lists {row.miRNA} as targeting {row["Target Gene"]}.'
        add_sample(out, make('mirna_to_target_gene','miRTarBase_microRNA_target_interaction.parquet',int(idx),user,row['Target Gene'],ev,{'mirna':row.miRNA}))
    return out

def build_gtex(limit=8000):
    df=pd.read_parquet(DL/'gtex_tissue_gene_tpm.parquet')
    for c in df.columns: df[c]=df[c].map(clean)
    df=df[(df['Gene']!='') & (df['Tissue']!='') & (df['Expression']!='')].drop_duplicates(subset=['Gene','Tissue'])
    df=sample_df(df,limit)
    out=[]
    for idx,row in df.iterrows():
        user=(f'Return the gene symbol for this GTEx tissue expression record.\nEvidence:\n- Ensembl description: {row.Description}\n- Tissue: {row.Tissue}\n- Expression TPM: {row.Expression}\n- Gene: {row.Gene}')
        ev=f'The GTEx record reports gene {row.Gene} expression in {row.Tissue}.'
        add_sample(out, make('gtex_expression_gene','gtex_tissue_gene_tpm.parquet',int(idx),user,row.Gene,ev,{'tissue':row.Tissue,'expression':row.Expression}))
    return out

def build_depmap(limit_each=3000):
    out=[]
    for fname, task, label in [('DepMap_CRISPRGeneEffect.csv','depmap_gene_effect','gene effect'),('DepMap_CRISPRGeneDependency.csv','depmap_gene_dependency','dependency probability')]:
        df=pd.read_csv(DL/fname, nrows=180, low_memory=False)
        id_col=df.columns[0]
        gene_cols=[c for c in df.columns[1:] if re.match(r'.+ \(\d+\)$', str(c))]
        pairs=[]
        for ridx,row in df.iterrows():
            cell=clean(row[id_col])
            vals=[]
            for c in gene_cols:
                try: vals.append((float(row[c]), c))
                except Exception: pass
            if not vals: continue
            vals_sorted=sorted(vals, key=lambda x:x[0])[:12] + sorted(vals, key=lambda x:x[0], reverse=True)[:8]
            for val,c in vals_sorted:
                gene=re.sub(r' \(\d+\)$','',c)
                pairs.append((ridx,cell,gene,val))
        random.shuffle(pairs)
        for ridx,cell,gene,val in pairs[:limit_each]:
            user=(f'Identify the gene from this DepMap CRISPR {label} record.\nEvidence:\n- Cell line model ID: {cell}\n- Gene: {gene}\n- Score: {val:.4g}\n\nReturn the gene symbol.')
            ev=f'In {fname}, {gene} has a {label} score of {val:.4g} in model {cell}.'
            add_sample(out, make(task,fname,int(ridx),user,gene,ev,{'cell_line':cell,'score':val}))
    return out

def main():
    builders=[build_gwas, build_variant, build_gene_info, build_disgenet, build_omim, build_proteinatlas, build_broad, build_ddinter, build_interactions, build_genesets, build_mirna, build_gtex, build_depmap]
    samples=[]
    source_counts={}
    for b in builders:
        part=b(); samples.extend(part); source_counts[b.__name__]=len(part); print(b.__name__, len(part), flush=True)
    # Deduplicate by user and assistant answer.
    seen=set(); dedup=[]
    for s in samples:
        key=hashlib.sha1((s['messages'][1]['content']+'\n'+s['answer']).encode('utf-8')).hexdigest()
        if key not in seen:
            seen.add(key); dedup.append(s)
    random.shuffle(dedup)
    split=int(len(dedup)*TRAIN_RATIO)
    train, val=dedup[:split], dedup[split:]
    for name,data in [('d1_datalake_all.jsonl',dedup),('d1_datalake_train.jsonl',train),('d1_datalake_val.jsonl',val)]:
        with (OUT/name).open('w',encoding='utf-8') as f:
            for s in data: f.write(json.dumps(s,ensure_ascii=False)+'\n')
    task_counts=Counter(s['task_type'] for s in dedup)
    file_counts=Counter(s['source_file'] for s in dedup)
    answer_types=Counter()
    for s in dedup:
        a=s['answer']
        if re.fullmatch(r'rs\d+',a,re.I): answer_types['rsid']+=1
        elif re.fullmatch(r'[A-Z][A-Z0-9-]{1,20}',a): answer_types['gene_symbol']+=1
        elif a.startswith('MIM '): answer_types['mim_text']+=1
        elif ',' in a: answer_types['list_text']+=1
        else: answer_types['free_text']+=1
    summary={'random_seed':SEED,'train_ratio':TRAIN_RATIO,'total_samples':len(dedup),'train_samples':len(train),'val_samples':len(val),'builder_counts_before_dedup':source_counts,'task_type_counts':dict(task_counts.most_common()),'source_file_counts':dict(file_counts.most_common()),'answer_type_counts':dict(answer_types.most_common()),'output_files':{'all':str(OUT/'d1_datalake_all.jsonl'),'train':str(OUT/'d1_datalake_train.jsonl'),'val':str(OUT/'d1_datalake_val.jsonl')}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    readme=['# D1 Datalake SFT v2','',f'Total samples: {len(dedup)}',f'Train samples: {len(train)}',f'Validation samples: {len(val)}','','## Task Type Counts','']
    for k,v in task_counts.most_common(): readme.append(f'- `{k}`: {v}')
    readme += ['','## Source File Counts','']
    for k,v in file_counts.most_common(): readme.append(f'- `{k}`: {v}')
    readme += ['','## Design Notes','','- Built from high-priority datalake sources identified by `artifacts/datalake_inventory_20260427`.','- Each record includes `source_file`, `source_row_id`, `answer`, `metadata`, and `quality_flags`.','- Assistant outputs use concise evidence plus exactly one `FINAL ANSWER` line; no `<think>` tags are included.','- This is D1 domain SFT data, not final D2/D3 task-format or reward data.']
    (OUT/'README.md').write_text('\n'.join(readme)+'\n',encoding='utf-8')
    print('WROTE', OUT, len(dedup), flush=True)
    print(json.dumps(summary,indent=2,ensure_ascii=False)[:4000])
if __name__ == '__main__': main()

