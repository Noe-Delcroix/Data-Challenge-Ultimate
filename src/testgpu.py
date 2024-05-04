import tensorflow as tf

# Vérifie si TensorFlow est construit avec le support de CUDA
print("Built with CUDA:", tf.test.is_built_with_cuda())

# Si CUDA est disponible, affiche des informations supplémentaires
if tf.test.is_built_with_cuda():
    # Affiche la version de CUDA utilisée par TensorFlow
    cuda_version = tf.sysconfig.get_build_info().get("cuda_version", "N/A")
    print("CUDA version used by TensorFlow:", cuda_version)

    # Vérifie le nombre de GPU disponibles
    num_gpus = tf.config.list_physical_devices('GPU')
    if num_gpus:
        print(f"Nombre de GPU disponibles : {len(num_gpus)}")
        # Boucle à travers les GPU disponibles et les affiche
        for i, gpu in enumerate(num_gpus):
            print(f"GPU {i}: {gpu.name}")
    else:
        print("Aucun GPU disponible.")
else:
    print("TensorFlow n'utilise pas CUDA sur ce système.")