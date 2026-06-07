# Kubernetes scaling (local: kind / minikube)

Run **parallel training jobs** — one pod per hyperparameter/algorithm config — all writing
to one shared volume, with automated result collection. Targets a single-node local cluster;
each job runs the whole sim stack inside its container via the image entrypoint.

## Prereqs
- A local cluster: `kind create cluster` **or** `minikube start`
- The image reachable by the cluster. For kind: `kind load docker-image rkaushik97/turtlebot3-drl:cpu`.
  For minikube: `minikube image load rkaushik97/turtlebot3-drl:cpu` (or just let it pull from Docker Hub).

## Single job
```bash
kubectl apply -f k8s/00-namespace.yaml -f k8s/01-pvc.yaml -f k8s/02-train-job.yaml
kubectl -n drlnav logs -f job/drl-train-ddpg
```

## Parallel sweep
Edit the `CONFIGS` list in `sweep.sh` (one line per run: `name|algo|exp|DRL_* overrides`), then:
```bash
k8s/sweep.sh                 # launch all jobs (CPU image)
IMAGE_TAG=cuda k8s/sweep.sh  # GPU image; each pod requests one nvidia.com/gpu
kubectl -n drlnav get jobs -l app=drl-sweep -w
k8s/sweep.sh delete          # tear the sweep down
```
Each pod writes models under `model/<pod-host>/` and eval summaries under `results/` on the
shared `drl-results` PVC, so runs never collide.

## Collect results
```bash
k8s/collect-results.sh ./sweep-results   # copies the whole PVC to ./sweep-results
```
Spins a short-lived collector pod that mounts the PVC, `kubectl cp`s the tree out, deletes itself.

## Notes
- **PVC access mode** is `ReadWriteOnce` — fine here because a local cluster is single-node, so
  many pods share it on that node. A multi-node cloud cluster would need `ReadWriteMany` (NFS/EFS)
  or per-job volumes.
- **Resources:** Gazebo is heavy; each pod requests 2 CPU / 2Gi. Don't over-subscribe a laptop —
  keep the sweep small or raise node resources.
- **GPU sweep** needs the NVIDIA device plugin installed on the node(s).
