cd /mnt/nas100/forGPU2/Kimjihoo/3_project_HCCmove/batch
# → 작업 디렉토리를 batch 폴더로 이동

cat > install_repo_editables.sh <<'EOF'
# → 여기서부터 EOF까지의 내용을 파일 install_repo_editables.sh 로 저장
#   작은따옴표가 있는 'EOF' 이므로, 본문 안 변수/백슬래시 확장은 "그대로" 기록됨 (안전한 here-doc)

#!/usr/bin/env bash
# → 이 스크립트를 bash 해석기로 실행하게 하는 shebang

set -euo pipefail
# -e: 명령 실패(비0) 시 즉시 종료
# -u: 선언되지 않은 변수를 사용하면 오류
# -o pipefail: 파이프라인 중 하나라도 실패하면 전체 실패로 간주

REPO_ROOT="${1:-/workspace}"
# → 첫 번째 인자를 REPO_ROOT로 사용, 없으면 기본값 /workspace
#   (컨테이너에서 /mnt 가 /workspace 로 마운트된 전제를 반영)

echo "===> REPO_ROOT: ${REPO_ROOT}"
# → 어떤 디렉토리를 레포 루트로 삼는지 로그 출력

pip_install_editable () {
  local sub="$1"
  local path="${REPO_ROOT}/${sub}"
  [[ -d "$path" ]] || { echo "❌ Not found: $path"; exit 1; }
  echo "===> pip install -e $path"
  python -m pip install -e "$path" --no-cache-dir
}
# → 서브폴더(예: engine, cv 등)를 editable 모드(-e)로 설치하는 함수
#   서브폴더가 없으면 에러 후 종료

req_exist () { [[ -f "$1" ]] || { echo "❌ Requirement not found: $1"; exit 1; }; }
# → 파일 존재 여부 체크용 헬퍼 함수 (없으면 에러)

for d in engine cv pretrain pose det seg; do
  [[ -d "${REPO_ROOT}/${d}" ]] || { echo "❌ Not found: ${REPO_ROOT}/${d}"; exit 1; }
done
# → 필요한 6개 디렉토리(로컬 패키지)가 모두 있는지 사전 검증

pip_install_editable engine
# → engine 패키지를 editable(-e)로 설치

pip_install_editable cv
# → cv 패키지 설치

req_exist "${REPO_ROOT}/cv/requirements/optional.txt"
# → cv의 optional requirements 파일이 있는지 확인 (없으면 종료)

python -m pip install -r "${REPO_ROOT}/cv/requirements/optional.txt" --no-cache-dir
# → optional 의존성 일괄 설치

pip_install_editable pretrain
pip_install_editable pose
pip_install_editable det
pip_install_editable seg
# → 나머지 pretrain/pose/det/seg를 editable로 설치

echo "✅ Editable installs done."
# → 완료 로그

EOF
# → here-doc 끝. 위 본문이 파일로 저장 완료

chmod +x install_repo_editables.sh
# → 실행 권한 부여 (직접 ./install_repo_editables.sh 로 실행 가능하게)
