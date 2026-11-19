# Guía: extracción de tsfresh en AWS

Extrae las features de tsfresh de los 134 datasets en una máquina de AWS que **se configura, trabaja y se borra sola**. No hace falta conectarse por SSH. Las features quedan en un bucket S3 y de ahí pasan a Google Drive para que Colab corra TabPFN.

- **Instancia:** `c7a.8xlarge` (32 vCPU, 64 GB de RAM), región **Ohio (us-east-2)**. Si no está disponible, `c6i.8xlarge`.
- **Tiempo estimado:** ~10–15 h con `c7a.8xlarge` (~20–25 h con `c6i.8xlarge`). EigenWorms es ~80 % del total y va al final.
- **Costo estimado:** ~$1.6/h → **~$16–25 del crédito**. Tope duro: la máquina se apaga a las 30 h pase lo que pase.
- **Script de arranque:** [`scripts/aws/tsfresh_user_data.sh`](../scripts/aws/tsfresh_user_data.sh).

> Antes de empezar: confirma en *Billing → Credits* que el crédito cubre EC2, y que la cuenta puede usarse para este proyecto (decisión #8).

## Paso 1. Alerta de presupuesto (5 min)

1. Buscador de la consola → **Budgets** → **Create budget**.
2. *Use a template* → **Monthly cost budget**.
3. Nombre `ts2tabpfn`, monto **40 USD**, tu correo como destinatario → **Create budget**.

AWS te avisará al 85 % y al 100 %. Es solo una alarma: no detiene nada por sí sola.

## Paso 2. Bucket S3 (3 min)

1. Buscador → **S3** → **Create bucket**.
2. Región: **us-east-2 (Ohio)**, la misma que la instancia.
3. Nombre único en todo AWS, en minúsculas, por ejemplo `ts2tabpfn-tsfresh-` seguido de 6 números al azar. **Anótalo.**
4. Deja todo lo demás por defecto (acceso público bloqueado) → **Create bucket**.

Opcional, para no depender de Google Drive: sube ahí el zip de datos (`data/raw/tsc_download/tsc_data.zip` de tu laptop, 1.3 GB) con el nombre exacto `tsc_data.zip`. Si no lo subes, la máquina lo descarga de Google Drive con `gdown`.

## Paso 3. Rol para que la máquina escriba en el bucket (5 min)

1. Buscador → **IAM** → **Roles** → **Create role**.
2. *Trusted entity:* **AWS service** → *Use case:* **EC2** → **Next**.
3. *Permissions:* busca y marca **AmazonS3FullAccess** → **Next**.
4. Nombre: `ts2tabpfn-ec2-s3` → **Create role**.

## Paso 4. Lanzar la instancia (10 min)

Buscador → **EC2** → arriba a la derecha verifica **Ohio** → **Launch instance**.

| Campo | Valor |
|---|---|
| **Name** | `ts2tabpfn-tsfresh` |
| **AMI** | **Amazon Linux 2023** (la que aparece por defecto; trae `aws` instalado) |
| **Instance type** | **`c7a.8xlarge`** (si no aparece, `c6i.8xlarge`) |
| **Key pair** | **Proceed without a key pair** (no vamos a entrar por SSH) |
| **Network settings → Edit** | *Allow SSH traffic*: **desmarcado**. Sin reglas de entrada |
| **Configure storage** | **40 GiB**, gp3 |
| **Advanced details → IAM instance profile** | `ts2tabpfn-ec2-s3` |
| **Advanced details → Shutdown behavior** | **Terminate** |
| **Advanced details → Termination protection** | Disable |
| **Advanced details → User data** | Pega **todo** el contenido de `scripts/aws/tsfresh_user_data.sh` y **cambia la línea `BUCKET=`** por el nombre de tu bucket |

Revisa el resumen de la derecha (1 instancia, `c7a.8xlarge`) → **Launch instance**.

Errores posibles al lanzar:
- *"You have requested more vCPU capacity than your current vCPU limit of 32"*: tienes otra instancia encendida. Las 4 `t3.micro` de tu otro proyecto deben estar **detenidas**.
- *"Unsupported"* o *"Insufficient capacity"*: cambia a `c6i.8xlarge` o prueba otra zona (*Network settings → Subnet*).

## Paso 5. Seguir el avance (sin entrar a la máquina)

- **S3 → tu bucket → `logs/tsfresh.log`** → *Download* o *Open*. Se actualiza cada 10 minutos. Verás líneas como `[12/134] ECG200: extracting` y `(200, 777) in 3s`.
- **S3 → tu bucket → `features/tsfresh/`**: aparece un `.npz` por dataset terminado.
- **EC2 → Instances**: mientras trabaja, *Running*; al terminar, *Terminated* sola.

La primera actualización del log llega ~15 minutos después de lanzar (instalación + descarga de datos).

## Paso 6. Al terminar

1. Verifica en **EC2 → Instances** que esté **Terminated**. Si sigue *Running* después de 30 h, selecciónala → *Instance state* → **Terminate instance**.
2. En el bucket aparece **`tsfresh_features.zip`** (todas las features juntas). Descárgalo.
3. Revisa el final de `logs/tsfresh.log`: debe decir `Finished. Failed: none`. Si algún dataset falló, avísame.
4. Sube `tsfresh_features.zip` a **`MyDrive/ts2tabpfn/features/`** (en drive.google.com, con la cuenta de Colab).
5. En Colab, agrega y ejecuta una vez esta celda (después de montar Drive):
   ```python
   !unzip -o -q /content/drive/MyDrive/ts2tabpfn/features/tsfresh_features.zip -d /content/drive/MyDrive/ts2tabpfn/features/
   !ls /content/drive/MyDrive/ts2tabpfn/features/tsfresh | wc -l   # debe decir 134
   ```
6. Con eso, la sesión de Colab puede correr `EXTRACTOR = "tsfresh"`.

Cuando ya no lo necesites, puedes vaciar y borrar el bucket (el almacenamiento de ~2 GB cuesta centavos al mes).

## Si se corta a la mitad

El script es reanudable: lanza otra instancia igual (paso 4). Al arrancar, baja del bucket las features ya hechas y solo extrae las que faltan.

## Más rápido (opcional)

Con 64 vCPU (`c7a.16xlarge`) tardaría ~5–8 h por un costo total parecido. Requiere pedir en **Service Quotas → Amazon EC2 → "Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances"** un aumento a **64**. No hace falta esperarlo: la `c7a.8xlarge` entra en tu cuota actual.
