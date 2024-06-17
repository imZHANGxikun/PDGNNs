echo corafull CustomDecoupledAPPNP 
echo -------------------------
python train.py --dataset CoraFull-CL \
       --method TEM \
       --gpu 0 \
       --backbone CustomDecoupledAPPNP \
       --ILmode taskIL \
       --overwrite_result True

echo -------------------------

echo arxiv CustomDecoupledAPPNP 
echo -------------------------
python train.py --dataset Arxiv-CL \
       --method TEM \
       --gpu 0 \
       --backbone CustomDecoupledAPPNP \
       --ILmode taskIL \
       --overwrite_result True

echo -------------------------

echo reddit CustomDecoupledAPPNP 
echo -------------------------
python train.py --dataset Reddit-CL \
       --method TEM \
       --gpu 0 \
       --backbone CustomDecoupledAPPNP \
       --ILmode taskIL \
       --overwrite_result True

echo -------------------------

echo products CustomDecoupledAPPNP 
echo -------------------------
python train.py --dataset Products-CL \
       --method TEM \
       --gpu 0 \
       --backbone CustomDecoupledAPPNP \
       --ILmode taskIL \
       --overwrite_result True

echo -------------------------

echo corafull CustomDecoupledSGC 
echo -------------------------
python train.py --dataset CoraFull-CL \
       --method TEM \
       --gpu 0 \
       --backbone CustomDecoupledSGC \
       --ILmode taskIL \
       --overwrite_result True

echo -------------------------

echo arxiv CustomDecoupledSGC 
echo -------------------------
python train.py --dataset Arxiv-CL \
       --method TEM \
       --gpu 0 \
       --backbone CustomDecoupledSGC \
       --ILmode taskIL \
       --overwrite_result True

echo -------------------------

echo reddit CustomDecoupledSGC 
echo -------------------------
python train.py --dataset Reddit-CL \
       --method TEM \
       --gpu 0 \
       --backbone CustomDecoupledSGC \
       --ILmode taskIL \
       --overwrite_result True

echo -------------------------

echo products CustomDecoupledSGC 
echo -------------------------
python train.py --dataset Products-CL \
       --method TEM \
       --gpu 0 \
       --backbone CustomDecoupledSGC \
       --ILmode taskIL \
       --overwrite_result True

echo corafull CustomDecoupledS2GC 
echo -------------------------
python train.py --dataset CoraFull-CL \
       --method TEM \
       --gpu 1 \
       --backbone CustomDecoupledS2GC \
       --ILmode taskIL \
       --overwrite_result True

echo -------------------------

echo arxiv CustomDecoupledS2GC 
echo -------------------------
python train.py --dataset Arxiv-CL \
       --method TEM \
       --gpu 1 \
       --backbone CustomDecoupledS2GC \
       --ILmode taskIL \
       --overwrite_result True

echo -------------------------

echo reddit CustomDecoupledS2GC 
echo -------------------------
python train.py --dataset Reddit-CL \
       --method TEM \
       --gpu 1 \
       --backbone CustomDecoupledS2GC \
       --ILmode taskIL \
       --overwrite_result True

echo -------------------------

echo products CustomDecoupledS2GC 
echo -------------------------
python train.py --dataset Products-CL \
       --method TEM \
       --gpu 1 \
       --backbone CustomDecoupledS2GC \
       --ILmode taskIL \
       --overwrite_result True



