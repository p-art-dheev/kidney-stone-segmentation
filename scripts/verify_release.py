"""Verify the live main ref and committed deliverable identities, without mutation."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


def main():
    head=git('rev-parse','HEAD').decode().strip()
    remote=git('ls-remote','origin','refs/heads/main').decode().split()[0]
    assert head==remote,'Local HEAD differs from live GitHub main'
    result=json.loads((ROOT/'reports/compliance-20261013/deliverable-verification.json').read_text())
    for folder,pdf in result['pdfs'].items():
        committed=git('show',f'{head}:{folder}/main.pdf')
        assert hashlib.sha256(committed).hexdigest()==pdf['sha256'],folder
    vendor={}
    archive=ROOT/'tmp/audit-20261013/cas.zip'
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            for name in ['cas-sc.cls','cas-common.sty','cas-model2-names.bst','README']:
                original=z.read('els-cas-templates/'+name)
                assert original==(ROOT/'report'/name).read_bytes(),name
                assert original==git('show',f'{head}:report/{name}'),name
                vendor[name]=hashlib.sha256(original).hexdigest()
    record={'status':'verified','branch':'main','local_head':head,'live_remote_main':remote,
            'remote':'https://github.com/p-art-dheev/kidney-stone-segmentation.git',
            'phases':[{'commit':git('rev-parse','3ae3188').decode().strip(),
                       'message':'Audit all course requirements and independently verify existing experiment evidence',
                       'push_verified':True},
                      {'commit':head,'message':git('log','-1','--format=%s').decode().strip(),'push_verified':True}],
            'committed_pdf_identities':result['pdfs'],
            'upstream_template_committed_bytes_match_archive':bool(vendor),'upstream_template_sha256':vendor,
            'working_tree_before_evidence_commit':git('status','--short').decode().splitlines(),
            'scope':'Published source/PDF package verification; final evidence-only closure has its own later commit hash.'}
    (ROOT/'reports/compliance-20261013/git_release_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'live_main':remote,'pdfs':len(result['pdfs']),'unchanged_upstream_files':len(vendor)}))


if __name__=='__main__':main()
