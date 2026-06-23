#!/usr/bin/env bash
#SBATCH --job-name=goal3-tools-prep
#SBATCH --partition=share
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=goal-3/logs/%x-%j.out
#SBATCH --error=goal-3/logs/%x-%j.err

set -euo pipefail
umask 077

module purge
if ! command -v srun >/dev/null 2>&1; then
    module load slurm
fi
module load "gcc/${GOAL3_GCC_MODULE:-12.5}" || true

repo_root=${GOAL3_REPO_ROOT:-/nfs/hpc/share/peterj29/pg/src/pg}
script_dir="$repo_root/goal-3/scripts"
# shellcheck disable=SC1091
source "$script_dir/common.sh"

goal3_init_run_dir
goal3_record_context "$GOAL3_RUN_DIR"

lrzip_prefix="$GOAL3_TOOLS_DIR/lrzip"

record_lrzip_version() {
    local lrzip_bin=$1
    if "$lrzip_bin" --version >"$GOAL3_RUN_DIR/lrzip-version.txt" 2>&1; then
        return 0
    fi
    if "$lrzip_bin" -V >"$GOAL3_RUN_DIR/lrzip-version.txt" 2>&1; then
        return 0
    fi
    "$lrzip_bin" --help >"$GOAL3_RUN_DIR/lrzip-version.txt" 2>&1
}

if [[ -x "$lrzip_prefix/bin/lrzip" ]]; then
    if record_lrzip_version "$lrzip_prefix/bin/lrzip"; then
        {
            echo "status=passed"
            echo "lrzip=$lrzip_prefix/bin/lrzip"
            cat "$GOAL3_RUN_DIR/lrzip-version.txt"
        } | tee "$GOAL3_RUN_DIR/status.txt"
        exit 0
    fi
    echo "Existing lrzip is not runnable; rebuilding it." >&2
fi

build_root="/scratch/$USER/${SLURM_JOB_ID:-manual}/goal3-tools"
mkdir -p "$build_root" "$GOAL3_TOOLS_DIR"

ensure_autoconf() {
    local need_local=1
    if command -v autoconf >/dev/null 2>&1; then
        local version
        version=$(autoconf --version | head -1 | awk '{print $NF}')
        if [[ "$(printf '2.71\n%s\n' "$version" | sort -V | head -1)" == "2.71" ]]; then
            need_local=0
        fi
    fi
    if [[ "$need_local" -eq 0 ]]; then
        return 0
    fi

    local ac_prefix="$GOAL3_TOOLS_DIR/autoconf-2.72"
    if [[ -x "$ac_prefix/bin/autoconf" ]]; then
        export PATH="$ac_prefix/bin:$PATH"
        return 0
    fi

    local ac_root="$build_root/autoconf-build"
    mkdir -p "$ac_root"
    cd "$ac_root"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL https://ftp.gnu.org/gnu/autoconf/autoconf-2.72.tar.xz \
            -o autoconf-2.72.tar.xz
    elif command -v wget >/dev/null 2>&1; then
        wget -O autoconf-2.72.tar.xz \
            https://ftp.gnu.org/gnu/autoconf/autoconf-2.72.tar.xz
    else
        echo "Neither curl nor wget is available for autoconf download." >&2
        exit 4
    fi
    tar -xf autoconf-2.72.tar.xz
    cd autoconf-2.72
    ./configure --prefix="$ac_prefix"
    make -j"${SLURM_CPUS_PER_TASK:-4}"
    make install
    export PATH="$ac_prefix/bin:$PATH"
}

ensure_lzo() {
    local lzo_prefix="$GOAL3_TOOLS_DIR/lzo-2.10"
    if [[ ! -f "$lzo_prefix/lib/liblzo2.so" && ! -f "$lzo_prefix/lib/liblzo2.a" ]]; then
        local lzo_root="$build_root/lzo-build"
        mkdir -p "$lzo_root"
        cd "$lzo_root"
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL https://www.oberhumer.com/opensource/lzo/download/lzo-2.10.tar.gz \
                -o lzo-2.10.tar.gz
        elif command -v wget >/dev/null 2>&1; then
            wget -O lzo-2.10.tar.gz \
                https://www.oberhumer.com/opensource/lzo/download/lzo-2.10.tar.gz
        else
            echo "Neither curl nor wget is available for lzo download." >&2
            exit 4
        fi
        tar -xf lzo-2.10.tar.gz
        cd lzo-2.10
        ./configure --prefix="$lzo_prefix"
        make -j"${SLURM_CPUS_PER_TASK:-4}"
        make install
    fi

    export CPPFLAGS="-I$lzo_prefix/include ${CPPFLAGS:-}"
    export LDFLAGS="-L$lzo_prefix/lib -Wl,-rpath,$lzo_prefix/lib ${LDFLAGS:-}"
    export LD_LIBRARY_PATH="$lzo_prefix/lib:${LD_LIBRARY_PATH:-}"
    export PKG_CONFIG_PATH="$lzo_prefix/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
}

ensure_lz4() {
    local lz4_version=${GOAL3_LZ4_VERSION:-1.9.4}
    local lz4_prefix="$GOAL3_TOOLS_DIR/lz4-$lz4_version"
    if [[ ! -f "$lz4_prefix/lib/liblz4.so" && ! -f "$lz4_prefix/lib/liblz4.a" ]]; then
        local lz4_root="$build_root/lz4-build"
        mkdir -p "$lz4_root"
        cd "$lz4_root"
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL "https://github.com/lz4/lz4/archive/refs/tags/v${lz4_version}.tar.gz" \
                -o "lz4-${lz4_version}.tar.gz"
        elif command -v wget >/dev/null 2>&1; then
            wget -O "lz4-${lz4_version}.tar.gz" \
                "https://github.com/lz4/lz4/archive/refs/tags/v${lz4_version}.tar.gz"
        else
            echo "Neither curl nor wget is available for lz4 download." >&2
            exit 4
        fi
        tar -xf "lz4-${lz4_version}.tar.gz"
        cd "lz4-$lz4_version"
        make -j"${SLURM_CPUS_PER_TASK:-4}"
        make PREFIX="$lz4_prefix" install
    fi

    export CPPFLAGS="-I$lz4_prefix/include ${CPPFLAGS:-}"
    export LDFLAGS="-L$lz4_prefix/lib -Wl,-rpath,$lz4_prefix/lib ${LDFLAGS:-}"
    export LD_LIBRARY_PATH="$lz4_prefix/lib:${LD_LIBRARY_PATH:-}"
    export PKG_CONFIG_PATH="$lz4_prefix/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
}

cd "$build_root"
lrzip_version=${GOAL3_LRZIP_VERSION:-0.651}
lrzip_url=${GOAL3_LRZIP_TARBALL_URL:-https://github.com/ckolivas/lrzip/releases/download/v${lrzip_version}/lrzip-${lrzip_version}.tar.xz}

downloaded=0
if command -v curl >/dev/null 2>&1; then
    if curl -fsSL "$lrzip_url" -o lrzip.tar.xz; then
        downloaded=1
    fi
elif command -v wget >/dev/null 2>&1; then
    if wget -O lrzip.tar.xz "$lrzip_url"; then
        downloaded=1
    fi
else
    echo "Neither curl nor wget is available for lrzip release download." >&2
    exit 4
fi

if [[ "$downloaded" -eq 1 ]] && tar -xf lrzip.tar.xz; then
    src_dir=$(find "$build_root" -maxdepth 1 -type d -name 'lrzip*' | head -1)
else
    echo "Release tarball failed; falling back to git clone." >&2
    git clone --depth 1 --branch "v${lrzip_version}" https://github.com/ckolivas/lrzip.git
    src_dir="$build_root/lrzip"
fi

if [[ -z "${src_dir:-}" || ! -d "$src_dir" ]]; then
    echo "Could not locate lrzip source directory." >&2
    exit 5
fi

cd "$src_dir"

ensure_lzo
ensure_lz4
cd "$src_dir"

if [[ -x ./configure ]]; then
    :
elif [[ -x ./autogen.sh ]]; then
    ensure_autoconf
    ./autogen.sh || autoreconf -fi
else
    ensure_autoconf
    autoreconf -fi
fi

cd "$src_dir"
./configure --prefix="$lrzip_prefix"
make -j"${SLURM_CPUS_PER_TASK:-4}"
make install

if [[ ! -x "$lrzip_prefix/bin/lrzip" ]]; then
    echo "lrzip install did not create $lrzip_prefix/bin/lrzip" >&2
    exit 6
fi

if ! record_lrzip_version "$lrzip_prefix/bin/lrzip"; then
    echo "Installed lrzip exists but is not runnable." >&2
    cat "$GOAL3_RUN_DIR/lrzip-version.txt" >&2 || true
    exit 7
fi

{
    echo "status=passed"
    echo "lrzip=$lrzip_prefix/bin/lrzip"
    cat "$GOAL3_RUN_DIR/lrzip-version.txt"
} | tee "$GOAL3_RUN_DIR/status.txt"
