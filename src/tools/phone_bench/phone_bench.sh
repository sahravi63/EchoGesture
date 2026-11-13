#!/system/bin/sh
# EchoGesture on-phone measurement (runs ON the phone; started over adb, then the cable is unplugged).
#   1. classifier latency probe on a cool phone
#   2. 300 s idle baseline        (battery charge counter, voltage and temperatures every 5 s)
#   3. 600 s pipeline load: MediaPipe Holistic's per-frame models at 15 fps (pose, face, 2 hands)
#      + the 3 INT8 classifiers x 2 views once per second, all with LiteRT benchmark_model
#   4. classifier latency probe on the warm phone (thermal drift)
# Results: /data/local/tmp/eg/out/   (pull them with adb when the cable is back)
D=/data/local/tmp/eg
B=$D/benchmark_model
O=$D/out
rm -rf $O; mkdir -p $O
cd $D
getprop ro.product.model > $O/device.txt
getprop ro.soc.model >> $O/device.txt
# Battery sysfs is not readable from adb shell on many phones (Xiaomi/MediaTek), but the
# fuel gauge's coulomb counter is: dumpsys battery "Charge counter" (uAh). Average power over
# a phase = drop in charge / time x mean voltage.
echo "phase,t,charge_uAh,voltage_mV,batt_temp_dC,cpu_temp_C,plugged" > $O/power.csv
log() {
  b=$(dumpsys battery)
  q=$(echo "$b" | grep "Charge counter" | tr -dc 0-9)
  v=$(echo "$b" | grep "^  voltage:" | tr -dc 0-9)
  t=$(echo "$b" | grep "^  temperature:" | tr -dc 0-9)
  p=$(echo "$b" | grep -c "powered: true")
  c=$(dumpsys thermalservice | sed -n '/Current temperatures from HAL/,/cooling/p' | grep "mName=CPU" | head -1 | sed 's/.*mValue=\([0-9.]*\).*/\1/')
  echo "$1,$(date +%s),$q,$v,$t,$c,$p" >> $O/power.csv
}
probe() { $B --graph=$D/echogesture_int8_s1.tflite --num_threads=4 --warmup_runs=20 --num_runs=300 > $O/lat_$1.txt 2>&1; }

echo "unplug the USB cable now (starting in 45 s)"; sleep 45
probe start
i=0; while [ $i -lt 60 ]; do log idle; sleep 5; i=$((i+1)); done

run() { $B --graph=$D/$1 --num_threads=$2 --run_delay=$3 --num_runs=1000000 --max_secs=600 --warmup_runs=1 > $O/load_$4.txt 2>&1 & }
run pose_landmarks_detector.tflite 1 0.0667 pose          # 15 fps
run face_landmarks_detector.tflite 1 0.0667 face          # 15 fps
run hand_landmarks_detector.tflite 1 0.0333 hands         # 2 hands x 15 fps
run hand_roi_refinement.tflite 1 0.0333 handroi           # 2 hands x 15 fps
run echogesture_int8_s1.tflite 4 0.5 cls1                 # 3 models x 2 views = 6 calls per sign,
run echogesture_int8_s2.tflite 4 0.5 cls2                 #   one sign per second (busier than real
run echogesture_int8_s3.tflite 4 0.5 cls3                 #   signing, so an upper bound)
i=0; while [ $i -lt 120 ]; do log load; sleep 5; i=$((i+1)); done
wait
probe end
log after
echo done > $O/DONE
