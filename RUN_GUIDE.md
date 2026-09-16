# Hướng dẫn chạy UniAV (WSL2 → Colab)

Tài liệu này mô tả cách chạy **train** và **eval** của UniAV với dữ liệu DESED bạn đang có,
trên hai môi trường:

- **WSL2 (Ubuntu 24.04)** — chỉ để *smoke-test* pipeline (máy bạn dùng GPU AMD Radeon, không có CUDA → chạy CPU).
- **Google Colab** — để train/eval thật trên GPU NVIDIA.

Code trong repo đã được sửa một số lỗi tương thích (xem [mục 6](#6-những-thay-đổi-đã-áp-dụng-vào-code)).
Tất cả thay đổi đang nằm ở working tree, **chưa commit**, bạn có thể xem bằng `git diff`.

---

## 1. Tóm tắt dữ liệu bạn đang có

Mình đã kiểm tra chéo annotation và feature — dữ liệu **đầy đủ và khớp nhau 100%**:

| Mục | Giá trị |
|---|---|
| `dcase_all.json` | 1539 clip, 5936 segment, 10 lớp (label_id 0–9) |
| Split `validation` (dùng để **train**) | 975 clip |
| Split `public_eval` (dùng để **eval/test**) | 564 clip |
| `desed_onepeace_audio_features.tar.gz` | 1539 file `<id>_one_peace_audio.npy` |
| `desed_onepeace_visual_features.tar.gz` | 1539 file `<id>_one_peace_video_finetune.npy` |
| Shape mỗi feature | `(T, 1536)`, T ≈ 37–40 (clip 10s, stride 0.25s) |
| Số clip thiếu feature | **0** |

Nói cách khác: đây không phải "một phần nhỏ", mà là **toàn bộ dataset DESED** mà paper dùng cho TASK3.
Bạn train/eval được TASK3 hoàn chỉnh. Chỉ thiếu ActivityNet (TASK1) và UnAV-100 (TASK2).

> **Hệ quả quan trọng:** chỉ chạy được `--tasks 3`.
> Nếu chạy `--tasks 1-2-3` sẽ lỗi `assert os.path.exists(feat_folder)` vì thiếu 2 dataset kia.
> Tuy vậy **model vẫn luôn được dựng với đủ 3 head** (vì `configs/*.yaml` khai báo cả TASK1/2/3),
> nên checkpoint pretrain `uni_model_epoch_006.pth.tar` vẫn load được bình thường.

---

## 2. Bố cục thư mục cần tạo

Code đọc dữ liệu theo đường dẫn **tương đối từ thư mục chạy lệnh** (`UniAV-fixed/`):

```
UniAV-fixed/
├── configs/multi_task_anet_unav_dcase.yaml
├── train.py, eval.py
├── libs/
└── data/
    ├── activitynet13/anet_prompt.npy        ← đã có sẵn trong repo
    ├── unav100/unav100_prompt.npy           ← đã có sẵn trong repo
    └── desed/
        ├── dcase_prompt.npy                 ← đã có sẵn trong repo
        ├── annotations/
        │   └── dcase_all.json               ← BẠN PHẢI COPY VÀO
        └── av_features/
            ├── <id>_one_peace_audio.npy           ← GIẢI NÉN VÀO ĐÂY
            └── <id>_one_peace_video_finetune.npy  ← (3078 file, phẳng, không thư mục con)
```

Lưu ý: 2 file `.tar.gz` giải nén ra thư mục con `onepeace_finetune_features_025s/`,
nên phải dùng `--strip-components=1` để đổ file trực tiếp vào `av_features/`.

Lệnh chuẩn bị (chạy trong `UniAV-fixed/`):

```bash
mkdir -p data/desed/annotations data/desed/av_features

SRC=/mnt/d/Học/KL/Code/UniAV/data/desed      # WSL; trên Colab đổi đường dẫn
cp "$SRC/dcase_all.json" data/desed/annotations/

tar -xzf "$SRC/desed_onepeace_audio_features.tar.gz"  -C data/desed/av_features --strip-components=1
tar -xzf "$SRC/desed_onepeace_visual_features.tar.gz" -C data/desed/av_features --strip-components=1

ls data/desed/av_features | wc -l   # phải ra 3078
```

---

## 3. Chạy trên WSL2 (Ubuntu) — smoke-test CPU

### 3.1. Cài công cụ hệ thống (cần `sudo`, nhập mật khẩu WSL của bạn)

```bash
sudo apt update
sudo apt install -y build-essential python3-venv python3-dev
```

`build-essential` là **bắt buộc**: một phần NMS viết bằng C++ và phải biên dịch được (`g++`, `-fopenmp`).

### 3.2. Copy code vào ext4 (quan trọng cho tốc độ)

Đọc hàng nghìn file `.npy` qua `/mnt/d` (NTFS qua 9p) **chậm hơn ext4 nhiều lần**.
Nên copy hẳn project vào home của WSL:

```bash
mkdir -p ~/work && cp -r "/mnt/d/Học/KL/Code/UniAV/UniAV-fixed" ~/work/
cd ~/work/UniAV-fixed
```

### 3.3. Tạo virtualenv + cài thư viện (bản CPU)

Ubuntu 24.04 có PEP 668 nên **bắt buộc dùng venv**, đừng `pip install` thẳng vào system Python.

```bash
python3 -m venv ~/venvs/uniav
source ~/venvs/uniav/bin/activate
pip install --upgrade pip wheel setuptools

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas pyyaml joblib h5py tensorboardX
```

### 3.4. Biên dịch NMS 1D

```bash
cd libs/utils
pip install .            # nếu lỗi thì: python setup.py install
cd ../..
python -c "import nms_1d_cpu; print('nms ok')"
```

Phải in ra `nms ok`. Nếu không, `libs/utils/nms.py` sẽ `ImportError` ngay khi import `libs.utils`.

> Mỗi khi bạn đổi version PyTorch thì phải biên dịch lại bước này.

### 3.5. Chuẩn bị dữ liệu

Chạy khối lệnh ở [mục 2](#2-bố-cục-thư-mục-cần-tạo).

### 3.6. Tạo subset nhỏ để smoke-test (khuyến nghị)

Trên CPU, một epoch đầy đủ (60 iteration) + eval 564 clip sẽ rất lâu.
Tạo một subset ~40 clip nhưng vẫn phủ đủ 10 lớp:

```bash
python - <<'EOF'
import json, collections
src = 'data/desed/annotations/dcase_all.json'
db  = json.load(open(src))['database']
out, seen = {}, collections.Counter()
for subset, quota in (('validation', 32), ('public_eval', 16)):
    picked = 0
    # ưu tiên clip mang lớp còn thiếu, rồi lấy bừa cho đủ quota
    for need_new in (True, False):
        for k, v in db.items():
            if picked >= quota or k in out or v['subset'] != subset:
                continue
            labs = {a['label_id'] for a in v['annotations']}
            if need_new and not (labs - set(seen)):
                continue
            out[k] = v; seen.update(labs); picked += 1
print('clips:', len(out), '| classes covered:', len(seen))
json.dump({'database': out}, open('data/desed/annotations/dcase_smoke.json', 'w'))
EOF

sed 's|./data/desed/annotations/dcase_all.json|./data/desed/annotations/dcase_smoke.json|' \
    configs/multi_task_anet_unav_dcase.yaml > configs/smoke_desed.yaml
```

> mAP đo trên subset này **không có ý nghĩa khoa học** — nó chỉ để xác nhận pipeline chạy thông.

### 3.7. Chạy smoke-test

**Eval bằng checkpoint pretrain:**

```bash
python eval.py ./configs/smoke_desed.yaml \
    /mnt/d/Học/KL/Code/UniAV/uni_model_epoch_006.pth.tar \
    --tasks 3 --num_workers 0 --batch_size 4
```

**Train 1 epoch:**

```bash
python train.py ./configs/smoke_desed.yaml \
    --output smoke --tasks 3 --num_train_epochs 1 \
    --num_workers 0 --batch_size 4 -p 1 -c 0
```

Lưu ý về RAM: WSL của bạn có ~7 GB. File checkpoint 2 GB chứa 4 bản state
(`state_dict`, `state_dict_ema`, 2 moment của AdamW) nên `torch.load` sẽ ngốn ~2 GB.
Nếu bị OOM, bỏ qua eval-with-pretrained ở local và làm trực tiếp trên Colab.

---

## 4. Chạy trên Google Colab (GPU)

### Cell 1 — kiểm tra GPU & mount Drive

```python
!nvidia-smi
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2 — đưa code về máy local của Colab

```python
%cd /content
# đổi đường dẫn cho khớp Drive của bạn
!cp -r "/content/drive/MyDrive/UniAV/UniAV-fixed" /content/UniAV-fixed
%cd /content/UniAV-fixed
!ls
```

> **Đừng** train trực tiếp từ thư mục Drive. Google Drive FUSE rất chậm với hàng nghìn
> file `.npy` nhỏ; luôn giải nén dữ liệu ra `/content` (đĩa cục bộ của VM).

### Cell 3 — cài thư viện + biên dịch NMS

```python
# Colab đã có sẵn torch (CUDA), numpy, pandas, pyyaml, joblib, h5py
!pip install -q tensorboardX
!cd libs/utils && pip install -q . && cd ../..
!python -c "import nms_1d_cpu, torch; print('nms ok |', torch.__version__, torch.cuda.is_available())"
```

Nếu gặp lỗi liên quan numpy 2.x: `!pip install -q "numpy<2"` rồi **Runtime → Restart session**
và chạy lại từ Cell 2.

### Cell 4 — chuẩn bị dữ liệu DESED

```python
SRC = "/content/drive/MyDrive/UniAV/data/desed"   # nơi bạn để 3 file dữ liệu trên Drive

!mkdir -p data/desed/annotations data/desed/av_features
!cp "{SRC}/dcase_all.json" data/desed/annotations/
!tar -xzf "{SRC}/desed_onepeace_audio_features.tar.gz"  -C data/desed/av_features --strip-components=1
!tar -xzf "{SRC}/desed_onepeace_visual_features.tar.gz" -C data/desed/av_features --strip-components=1
!ls data/desed/av_features | wc -l      # 3078
```

### Cell 5 — train

Code dùng `DistributedDataParallel`, nên cách chạy đúng với paper là qua `torchrun`
(1 process, 1 GPU):

```python
!OMP_NUM_THREADS=2 torchrun --nproc_per_node=1 --master_port=29501 \
    train.py ./configs/multi_task_anet_unav_dcase.yaml \
    --output desed_only --tasks 3 --num_train_epochs 5 \
    --num_workers 2 -p 10 -c 0 2>&1 | tee train_desed.log
```

Cũng có thể chạy **không DDP** (đơn giản hơn, tránh mọi vấn đề NCCL/port):

```python
!python train.py ./configs/multi_task_anet_unav_dcase.yaml \
    --output desed_only --tasks 3 --num_train_epochs 5 \
    --num_workers 2 -p 10 -c 0 2>&1 | tee train_desed.log
```

Hai cách cho checkpoint có tên tham số khác nhau (`module.*` vs không),
nhưng `eval.py` và `--resume` đều đã tự nhận diện và quy đổi.
Dù vậy, **nên chọn một cách và giữ nguyên** cho cả train lẫn resume.

Con số mong đợi với TASK3: `975 // 16 = 60` iteration/epoch, tổng
`5 + 2 (warmup) = 7` epoch. Checkpoint lưu tại `./ckpt/multi_task_anet_unav_dcase_desed_only/`.

### Cell 6 — eval

```python
CKPT = "./ckpt/multi_task_anet_unav_dcase_desed_only"   # hoặc trỏ thẳng file .pth.tar
!python eval.py ./configs/multi_task_anet_unav_dcase.yaml {CKPT} \
    --tasks 3 -p 10 2>&1 | tee eval_desed.log
```

Nếu đưa vào một **thư mục**, code lấy file `*.pth.tar` cuối cùng theo thứ tự alphabet —
lưu ý `best.pth.tar` sẽ đứng trước `epoch_006.pth.tar`. Muốn chắc chắn thì **trỏ thẳng file**:

```python
!python eval.py ./configs/multi_task_anet_unav_dcase.yaml \
    "/content/drive/MyDrive/UniAV/uni_model_epoch_006.pth.tar" --tasks 3
```

Log in ra mAP tại 9 ngưỡng tIoU (0.1 → 0.9) và Average mAP.

### Cell 7 — lưu kết quả về Drive & TensorBoard

```python
!cp -r ./ckpt/multi_task_anet_unav_dcase_desed_only "/content/drive/MyDrive/UniAV/ckpt_out"
```

```python
%load_ext tensorboard
%tensorboard --logdir ./ckpt/multi_task_anet_unav_dcase_desed_only/logs
```

### Khi Colab ngắt giữa chừng

```python
!python train.py ./configs/multi_task_anet_unav_dcase.yaml \
    --output desed_only --tasks 3 --num_train_epochs 5 \
    --resume ./ckpt/multi_task_anet_unav_dcase_desed_only/epoch_006.pth.tar
```

`--resume` khôi phục cả optimizer/scheduler và đặt `start_epoch = epoch + 1`.

> **Bẫy:** nếu bạn muốn *fine-tune từ checkpoint pretrain của tác giả* (`uni_model_epoch_006.pth.tar`,
> tức `epoch = 6`), thì `start_epoch = 7`. Với `--num_train_epochs 5` thì `max_epochs = 7`,
> nên vòng lặp `range(7, 7)` **rỗng — không train gì cả**.
> Muốn fine-tune thêm 8 epoch thì phải đặt `--num_train_epochs 13` (→ `max_epochs = 15`).

---

## 5. Giải thích tham số dòng lệnh

| Tham số | Ý nghĩa |
|---|---|
| `config` (vị trí 1) | file YAML. Luôn khai báo đủ TASK1/2/3 để dựng đúng kiến trúc model. |
| `ckpt` (vị trí 2, chỉ `eval.py`) | file `.pth.tar` hoặc thư mục chứa nó. |
| `--tasks` | `1`=ActivityNet/TAL, `2`=UnAV-100/AVEL, `3`=DESED/SED. Nối bằng `-`, vd `1-2-3`. Với dữ liệu hiện tại: **chỉ dùng `3`**. |
| `--num_train_epochs` | Số epoch *không kể* warmup. Tổng thực tế = giá trị này + `warmup_epochs` (mặc định 2). |
| `--output` | Hậu tố tên thư mục checkpoint: `./ckpt/<tên_config>_<output>`. |
| `--resume` | Đường dẫn checkpoint để train tiếp. |
| `-c/--ckpt-freq` | Lưu checkpoint mỗi N epoch. **`0` = chỉ lưu epoch cuối** (mỗi file ~2 GB, nên dùng `0` trên Colab). |
| `-p/--print-freq` | Tần suất in log. |
| `--num_workers` | *(mới)* ghi đè `cfg['num_workers']`. Dùng `0` khi debug. |
| `--batch_size` | *(mới)* ghi đè batch size của mọi task. |
| `-t/--topk` | *(chỉ `eval.py`)* giới hạn số segment đầu ra mỗi video. |

Các siêu tham số còn lại (lr, warmup, NMS, kiến trúc…) nằm ở
[libs/core/config.py](libs/core/config.py) và file YAML.

---

## 6. Những thay đổi đã áp dụng vào code

Repo gốc viết cho **Python 3.8 / PyTorch 1.11 / NumPy < 1.24**. Trên Colab (PyTorch 2.x, NumPy 2.x)
nó crash ngay. Dưới đây là toàn bộ chỉnh sửa, mỗi mục kèm lý do:

### Sửa lỗi tương thích (bắt buộc, nếu không thì không chạy được)

| File | Thay đổi | Lý do |
|---|---|---|
| [libs/utils/task_utils.py:4](libs/utils/task_utils.py#L4) | `from torch._six import inf` → `from math import inf` | `torch._six` bị xoá từ PyTorch 1.13. |
| [libs/utils/task_utils.py:121](libs/utils/task_utils.py#L121) | `iterator.next()` → `next(iterator)` | Method `.next()` của DataLoader iterator đã bị bỏ. |
| [libs/utils/metrics.py:301-302](libs/utils/metrics.py#L301-L302) | `.astype(np.float)` → `.astype(float)` | `np.float` bị xoá từ NumPy 1.24. |
| [train.py](train.py), [eval.py](eval.py) | `np.load('./data/dcase/dcase_prompt.npy')` → `load_prompt('./data/desed/...', './data/dcase/...')` | File thật nằm ở `data/desed/`, config cũng trỏ `data/desed/` — đường dẫn `data/dcase/` trong code là sai. |
| [train.py](train.py), [eval.py](eval.py) | `--local_rank` nhận thêm `--local-rank` và biến môi trường `LOCAL_RANK` | `torchrun` (PyTorch ≥ 2.0) không còn truyền cờ `--local_rank` nữa. |
| [train.py](train.py), [eval.py](eval.py) | `dist.get_rank()` → chỉ gọi khi `dist.is_initialized()` | Trước đây crash nếu chạy `python train.py` thẳng (không DDP). |
| [train.py](train.py), [eval.py](eval.py) | `map_location=lambda storage, loc: storage.cuda(...)` → `map_location=device` + helper `torch_load` | Cách cũ hỏng khi chạy CPU; thêm `weights_only=False` cho PyTorch ≥ 2.6. |
| [libs/utils/task_utils.py](libs/utils/task_utils.py), [libs/datasets/datasets.py](libs/datasets/datasets.py) | `persistent_workers=True` → `persistent_workers=(num_workers > 0)` | DataLoader báo lỗi khi `num_workers=0`. |

### Sửa lỗi logic

| File | Thay đổi | Lý do |
|---|---|---|
| [train.py](train.py) | Khối "lưu best checkpoint" được viết lại | Code gốc dựng dict `save_states` nhưng **không bao giờ ghi ra đĩa**, và ném `UnboundLocalError` nếu không task nào cải thiện ở lần eval đầu. Giờ ghi ra `best.pth.tar` mỗi khi có task cải thiện. |
| [libs/utils/train_utils.py](libs/utils/train_utils.py) — `make_optimizer` | Tiền tố `module.` được tự động phát hiện thay vì hard-code | Code gốc hard-code `no_decay.remove('module.cls_head.clip_proj.bias')`, nên **bắt buộc** phải bọc DDP; chạy `python train.py` thẳng sẽ `KeyError`. |
| [eval.py](eval.py), [train.py](train.py) | Thêm `match_ckpt_prefix()` | Cho phép checkpoint train bằng DDP (`module.*`) load vào model không-DDP và ngược lại. |

### Tiện ích thêm cho Colab

- `--num_workers` và `--batch_size` để ghi đè config từ dòng lệnh (RAM/VRAM Colab thay đổi theo phiên).
- `os.mkdir` → `os.makedirs(..., exist_ok=True)`: cho phép `--output` lồng nhiều cấp và chạy lại không lỗi.

**Chưa đụng tới:** thuật toán, loss, kiến trúc, siêu tham số — kết quả vẫn tái lập được như paper.

---

## 7. Xử lý sự cố

| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| `ModuleNotFoundError: No module named 'nms_1d_cpu'` | Chưa biên dịch NMS. Xem [3.4](#34-biên-dịch-nms-1d). Trên WSL nhớ `sudo apt install build-essential`. |
| `AssertionError` tại `assert os.path.exists(feat_folder)` | Sai bố cục `data/desed/`. Kiểm tra `ls data/desed/av_features | wc -l` = 3078. |
| `FileNotFoundError: none of these prompt files exist` | Thiếu file `*_prompt.npy` — chúng đi kèm repo, kiểm tra bạn đã copy cả thư mục `data/`. |
| `KeyError: 'module.cls_head.clip_proj.bias'` | Bạn đang chạy code **chưa được vá**. Xem [mục 6](#6-những-thay-đổi-đã-áp-dụng-vào-code). |
| `RuntimeError: Error(s) in loading state_dict ... Missing key(s)` | Checkpoint và model lệch số task. Config **phải** giữ đủ TASK1/2/3, dù `--tasks 3`. |
| `torchrun` treo / `Address already in use` | Đổi `--master_port` (vd 29502), hoặc chạy bằng `python train.py` không DDP. |
| Colab hết đĩa | Dùng `-c 0` (mỗi checkpoint ~2 GB). Xoá `ckpt/` cũ trước khi chạy lại. |
| Train rất chậm, GPU idle | Dữ liệu đang nằm trên Drive. Giải nén ra `/content` (xem Cell 4). |
| `RuntimeError: DataLoader worker ... killed` | Hết RAM → `--num_workers 0` và/hoặc `--batch_size 8`. |
| mAP = 0.00 ở mọi ngưỡng | Bình thường ở epoch đầu hoặc khi dùng subset smoke-test. Nếu kéo dài, kiểm tra `--tasks` khớp với dataset đang có. |
