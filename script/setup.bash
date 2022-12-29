#! /usr/bin/env bash

PROJECT_DIR=$(pwd)/..
WEIGHTS_DIR=$PROJECT_DIR/weights

REQS_TXT=$PROJECT_DIR/requirements.txt
RESNET_BASE_WEIGHT_URL="https://storage.googleapis.com/tensorflow/keras-applications/resnet/"
RESNET50_WEIGHTS_NOTOP="resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
RESNET50_WEIGHTS_NOTOP_MD5="4d473c1dd8becc155b73f8504c6f6626"
RESNET101_WEIGHTS_NOTOP="resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5"
RESNET101_WEIGHTS_NOTOP_MD5="88cf7a10940856eca736dc7b7e228a21"
RESNET50_WEIGHTS_NOTOP_URL=$RESNET_BASE_WEIGHT_URL$RESNET50_WEIGHTS_NOTOP
RESNET101_WEIGHTS_NOTOP_URL=$RESNET_BASE_WEIGHT_URL$RESNET101_WEIGHTS_NOTOP

printf "[INFO]: Upgrading pip...\n"
python3 -m pip install --upgrade pip
RETURN=$?
if [ $RETURN -ne 0 ]
then
  printf "[INFO]: Failed to upgrade pip. Quit.\n"
  exit $RETURN
fi

printf "[INFO]: Installing dependencies...\n"
python3 -m pip install -r $REQS_TXT
RETURN=$?
if [ $RETURN -ne 0 ]
then
  printf "[INFO]: Failed to install dependencies. Quit.\n"
  exit $RETURN
fi

printf "[INFO]: Creating weights directory...\n"
if [ ! -d "$WEIGHTS_DIR" ]
then
  mkdir $WEIGHTS_DIR
  RETURN=$?
  if [ $RETURN -ne 0 ]
  then
    printf "[INFO]: Failed to create weights directory. Quit.\n"
    exit $RETURN
  fi
else
  printf "[INFO]: weights directory exists. Skip.\n"
fi

printf "[INFO]: Downloading ResNet-50 pretrained weights (notop version)...\n"
if [ ! -f "$WEIGHTS_DIR/$RESNET50_WEIGHTS_NOTOP" ]
then
  cd $WEIGHTS_DIR
  curl -L --progress-bar $RESNET50_WEIGHTS_NOTOP_URL -o $RESNET50_WEIGHTS_NOTOP
  RETURN=$?
  if [ $RETURN -ne 0 ]
  then
    printf "[INFO]: Failed to download ResNet-50 pretrained weights. Quit.\n"
    exit $RETURN
  fi
else
  printf "[INFO]: ResNet-50 pretrained weights exists. Skip.\n"
fi

printf "[INFO]: Checking MD5 hash for ResNet-50 pretrained weights...\n"
OPENSSL_RESNET50_WEIGHTS_NOTOP=$(openssl dgst -md5 $WEIGHTS_DIR/$RESNET50_WEIGHTS_NOTOP)
RETURN=$?
if [ $RETURN -ne 0 ]
then
  printf "[INFO]: Failed to execute MD5(%s). Quit.\n" $RESNET50_WEIGHTS_NOTOP
  exit $RETURN
fi
MD5_RESNET50_WEIGHTS_NOTOP=$(cut -d " " -f 2 <<< $OPENSSL_RESNET50_WEIGHTS_NOTOP)
RETURN=$?
if [ $RETURN -ne 0 ]
then
  printf "[INFO]: Failed to execute cut %s. Quit.\n" $OPENSSL_RESNET50_WEIGHTS_NOTOP
  exit $RETURN
fi
if [ "$MD5_RESNET50_WEIGHTS_NOTOP" == "$RESNET50_WEIGHTS_NOTOP_MD5" ]
then
  printf "[INFO]: MD5(%s) check passed.\n" $RESNET50_WEIGHTS_NOTOP
else
  printf "[INFO]: MD5 hash doesn't match. ResNet-50 pretrained weights file incomplete or corrupted. Quit.\n"
  exit 1
fi

printf "[INFO]: Downloading ResNet-101 pretrained weights (notop version)...\n"
if [ ! -f "$WEIGHTS_DIR/$RESNET101_WEIGHTS_NOTOP" ]
then
  cd $WEIGHTS_DIR
  curl -L --progress-bar $RESNET101_WEIGHTS_NOTOP_URL -o $RESNET101_WEIGHTS_NOTOP
  RETURN=$?
  if [ $RETURN -ne 0 ]
  then
    printf "[INFO]: Failed to download ResNet-101 pretrained weights. Quit.\n"
    exit $RETURN
  fi
else
  printf "[INFO]: ResNet-101 pretrained weights exists. Skip.\n"
fi

printf "[INFO]: Checking MD5 hash for ResNet-101 pretrained weights...\n"
OPENSSL_RESNET101_WEIGHTS_NOTOP=$(openssl dgst -md5 $WEIGHTS_DIR/$RESNET101_WEIGHTS_NOTOP)
RETURN=$?
if [ $RETURN -ne 0 ]
then
  printf "[INFO]: Failed to execute MD5(%s). Quit.\n" $RESNET50_WEIGHTS_NOTOP
  exit $RETURN
fi
MD5_RESNET101_WEIGHTS_NOTOP=$(cut -d " " -f 2 <<< $OPENSSL_RESNET101_WEIGHTS_NOTOP)
RETURN=$?
if [ $RETURN -ne 0 ]
then
  printf "[INFO]: Failed to execute cut %s. Quit.\n" $OPENSSL_RESNET101_WEIGHTS_NOTOP
  exit $RETURN
fi
if [ "$MD5_RESNET101_WEIGHTS_NOTOP" == "$RESNET101_WEIGHTS_NOTOP_MD5" ]
then
  printf "[INFO]: MD5(%s) check passed.\n" $RESNET101_WEIGHTS_NOTOP
else
  printf "[ERROR]: MD5 hash doesn't match. ResNet-101 pretrained weights file incomplete or corrupted. Quit.\n"
  exit 1
fi

echo "[INFO]: Setup completed."