import unittest
import numpy as np
import pandas as pd
from neoantigen.metrics import ranking_metrics
from neoantigen.train.dataloader import make_splits, audit_splits, prepare_df
from neoantigen.benchmark.external import NetMHCpanBaseline


class EvaluationTests(unittest.TestCase):
    def test_allele_prior_has_no_within_allele_discrimination(self):
        y = np.array([1]*90+[0]*10+[1]*10+[0]*90)
        s = np.array([.9]*100+[.1]*100)
        m = ranking_metrics(y,s,np.array(['A']*100+['B']*100))
        self.assertGreater(m['auroc'],.8)
        self.assertEqual(m['within_allele_auroc'],.5)
        self.assertEqual(m['within_allele_n'],200)

    def test_ppv_cutoff_ties_and_nan_coverage(self):
        m = ranking_metrics([1]*50+[0]*100+[1], [1.]*150+[np.nan])
        self.assertAlmostEqual(m['ppv_at_100'],1/3)
        self.assertEqual(m['ppv_at_100_n'],100)
        self.assertEqual(m['n_scored'],150)
        self.assertEqual(ranking_metrics([1,1],[.1,.2])['auprc'],1.)
        self.assertEqual(ranking_metrics([0,0],[.1,.2])['ppv_at_100'],0.)
        self.assertNotIn('auroc', ranking_metrics([1,1],[.1,.2]))
        self.assertEqual(ranking_metrics([],[])['n_scored'],0)

    def test_peptide_default_split_drops_conflicts_and_duplicates(self):
        df = pd.DataFrame([dict(epitope=f'P{i}',mhc_sequence=f'M{j}',
                               mhc_allele=f'A{j}',y=i%2)
                           for i in range(30) for j in range(3)])
        df = pd.concat([df,df.iloc[:2],df.iloc[:1].assign(y=1)],ignore_index=True)
        parts = make_splits(df)
        self.assertEqual(sum(map(len,parts)),89)
        audit_splits(dict(zip(['train','valid','test'],parts)))
        self.assertFalse(any((d.epitope.eq('P0') & d.mhc_allele.eq('A0')).any() for d in parts))
        bad = dict(train=parts[0],valid=parts[1],test=pd.concat([parts[2],parts[0].iloc[:1]]))
        with self.assertRaisesRegex(ValueError,'share'):
            audit_splits(bad)

    def test_explicit_cedar_conflict_flags_are_excluded(self):
        df = pd.DataFrame(dict(peptide=['AAA','CCC','DDD','EEE'],
            mhc_amino_acid=['ACDE']*4, bind=[1,0,1,0],
            label_conflict=['True','False',1,0]))
        self.assertEqual(prepare_df(df).epitope.tolist(),['CCC','EEE'])

    def test_heldout_allele_also_holds_out_encoded_sequences(self):
        df = pd.DataFrame([dict(epitope=f"P{i}", mhc_sequence=f"M{j}",
             mhc_allele=f"A{j}", y=i%2) for i in range(4) for j in range(20)])
        df = pd.concat([df, df.loc[df.mhc_allele.eq('A0')].assign(mhc_allele='synonym')])
        parts = make_splits(df, split_group='allele')
        audit_splits(dict(zip(['train','valid','test'],parts)), 'allele')
        bad = {'train': df.loc[df.mhc_allele.eq('A0')],
               'valid': df.loc[df.mhc_allele.eq('A1')],
               'test': df.loc[df.mhc_allele.eq('synonym')]}
        with self.assertRaisesRegex(ValueError,'MHC sequences'):
            audit_splits(bad,'allele')

    def test_evaluation_checks_checkpoint_training_identity(self):
        import tempfile
        import json
        import hashlib
        from pathlib import Path
        from neoantigen.infer.evaluate import verify_unseen_alleles
        from neoantigen.provenance import file_sha256
        from neoantigen.train.dataloader import load_splits, DataConfig
        aa = 'ACDEFGHIKLMNPQRSTVWY'
        df = pd.DataFrame([dict(epitope='AA'+aa[i//20]+aa[i%20]+'AA',
            mhc_sequence='ACDEFG', mhc_allele='A0', y=i%2) for i in range(30)])
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            for frame,name in zip(make_splits(df),['train.csv','val.csv','test.csv']):
                frame.to_csv(tmp/name,index=False)
            train,_,_=load_splits(DataConfig(data_dir=str(tmp)))
            manifest=tmp/'run_manifest.json'
            manifest.write_text(json.dumps({'split_sha256':{'train':hashlib.sha256(train.to_csv(index=False).encode()).hexdigest()}}))
            ckpt={'args':{'data_dir':str(tmp),'split_group':'peptide'},
                  'run_manifest_sha256':file_sha256(manifest)}
            target=train.iloc[:1].assign(mhc_sequence='HIKLMN',mhc_allele='B0')
            self.assertTrue(verify_unseen_alleles(ckpt,tmp/'best.pt',target))
            with self.assertRaisesRegex(ValueError,'occurred in checkpoint training'):
                verify_unseen_alleles(ckpt,tmp/'best.pt',train.iloc[:1])
            changed=pd.read_csv(tmp/'train.csv')
            changed.loc[0,'y']=1-changed.loc[0,'y']
            changed.to_csv(tmp/'train.csv',index=False)
            with self.assertRaisesRegex(ValueError,'recorded hash'):
                verify_unseen_alleles(ckpt,tmp/'best.pt',target)

    def test_frozen_assignments_restore_order_and_require_complete_corpus(self):
        import tempfile
        from pathlib import Path
        from neoantigen.data.regroup import restore_assignments
        pool=pd.DataFrame(dict(peptide=['AAA','CCC','DDD','EEE'],
            mhc_allele=['A0','B0','C0','D0'],bind=[1,0,1,0]))
        assignments=pool[['peptide','mhc_allele']].copy()
        assignments['split']=['train','train','valid','test']
        assignments['split_row']=[1,0,0,0]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'assignments.csv.gz'
            assignments.to_csv(path,index=False)
            restored=restore_assignments(pool,path)
            self.assertEqual(restored['train'].peptide.tolist(),['CCC','AAA'])
            assignments.iloc[1:].to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'cover exactly'):
                restore_assignments(pool,path)

    def test_actual_netmhcpan42_header_and_identity(self):
        text = ''' Pos MHC Peptide Core Of Gp Gl Ip Il Icore Identity Score_EL %Rank_EL Score_BA %Rank_BA Aff(nM) BindLevel
1 HLA-A*02:01 AAAAAAAAA AAAAAAAAA 0 0 0 0 0 AAAAAAAAA PEPLIST 0.5 1.5 0.7 0.2 20 <= SB
2 HLA-A*02:01 CCCCCCCCC CCCCCCCCC 0 0 0 0 0 CCCCCCCCC PEPLIST 0.01 50 0.2 40 2000'''
        np.testing.assert_allclose(NetMHCpanBaseline._parse(text,2,['AAAAAAAAA','CCCCCCCCC']),[.985,.5])
        with self.assertRaises(ValueError):
            NetMHCpanBaseline._parse(text,2,['CCCCCCCCC','AAAAAAAAA'])


class AttentionTests(unittest.TestCase):
    def test_capture_preserves_predictions_and_removes_hooks(self):
        import torch
        from neoantigen.infer.explain import AttentionCapture
        from neoantigen.models.pooling import AttentionPool
        class Example(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = torch.nn.MultiheadAttention(4,2,batch_first=True)
                self.pool = AttentionPool(4)
            def forward(self,x):
                values,_ = self.attn(x,x,x)
                return self.pool(values,torch.ones(x.shape[:2]))
        model = Example().eval()
        x = torch.randn(1,3,4)
        expected = model(x)
        capture = AttentionCapture(model)
        torch.testing.assert_close(expected,model(x))
        self.assertEqual(capture.weights['attn'].shape,(1,3,3))
        np.testing.assert_allclose(capture.weights['pool'].sum(-1),1,rtol=1e-6)
        capture.close()
        self.assertFalse(model.attn._forward_hooks)

if __name__ == '__main__':
    unittest.main()
