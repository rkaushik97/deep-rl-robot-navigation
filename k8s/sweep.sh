#!/usr/bin/env bash
# Launch a PARALLEL hyperparameter sweep — one K8s Job (one pod) per config, all writing
# to the shared drl-results PVC under a pod-unique subdir. Edit the CONFIGS list below.
#
#   k8s/sweep.sh                 # apply all jobs in CONFIGS
#   IMAGE_TAG=cuda k8s/sweep.sh  # same, on the GPU image (adds an nvidia.com/gpu request)
#   k8s/sweep.sh delete          # tear the sweep down
#
# Each CONFIG line:  <name>|<algo>|<exp>|<space-separated DRL_* overrides>
# The DRL_* overrides win over the algo's config.sh inside the container (see entrypoint).
set -euo pipefail
NS=drlnav
IMAGE_TAG=${IMAGE_TAG:-cpu}
IMAGE=kaushik48/turtlebot3-drl:${IMAGE_TAG}
MAX_EPISODES=${DRL_MAX_EPISODES:-2000}

CONFIGS=(
  "ddpg-lr3e4|ddpg||DRL_LR=0.0003"
  "ddpg-lr1e3|ddpg||DRL_LR=0.001"
  "td3-base|td3||"
  "sac-rewardv|sac|reward_v_explore|"
)

# GPU image -> request one GPU per pod
GPU_RESOURCES=""
[ "$IMAGE_TAG" = "cuda" ] && GPU_RESOURCES=$'\n            limits:\n              nvidia.com/gpu: "1"'

if [ "${1:-apply}" = "delete" ]; then
  kubectl -n "$NS" delete jobs -l app=drl-sweep --ignore-not-found
  exit 0
fi

kubectl apply -f "$(dirname "$0")/00-namespace.yaml"
kubectl apply -f "$(dirname "$0")/01-pvc.yaml"

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME ALGO EXP OVERRIDES <<< "$cfg"
  JOB="drl-sweep-${NAME}"
  # build the env block (base + per-config DRL_* overrides)
  ENV_BLOCK=$(printf '            - name: %s\n              value: "%s"\n' \
                MODE train ALGO "$ALGO" EXP "$EXP" DRL_STAGE 9 DRL_MAX_EPISODES "$MAX_EPISODES")
  for kv in $OVERRIDES; do
    ENV_BLOCK+=$(printf '            - name: %s\n              value: "%s"\n' "${kv%%=*}" "${kv#*=}")
  done

  cat <<YAML | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB}
  namespace: ${NS}
  labels: { app: drl-sweep }
spec:
  backoffLimit: 1
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels: { app: drl-sweep, sweep-config: "${NAME}" }
    spec:
      restartPolicy: Never
      containers:
        - name: trainer
          image: ${IMAGE}
          imagePullPolicy: IfNotPresent
          env:
${ENV_BLOCK}
          resources:
            requests: { cpu: "2", memory: "2Gi" }${GPU_RESOURCES}
          volumeMounts:
            - { name: results, mountPath: /opt/drlnav/src/turtlebot3_drl/model, subPath: model }
            - { name: results, mountPath: /opt/drlnav/src/turtlebot3_drl/turtlebot3_drl/evaluation/results, subPath: results }
            - { name: dshm, mountPath: /dev/shm }
      volumes:
        - name: results
          persistentVolumeClaim: { claimName: drl-results }
        - name: dshm
          emptyDir: { medium: Memory, sizeLimit: 1Gi }
YAML
  echo "applied $JOB  (algo=$ALGO exp=${EXP:-base} overrides='${OVERRIDES}')"
done

echo
echo "watch:    kubectl -n $NS get jobs -l app=drl-sweep -w"
echo "logs:     kubectl -n $NS logs -f job/drl-sweep-<name>"
echo "collect:  k8s/collect-results.sh"
