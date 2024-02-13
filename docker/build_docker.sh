DOCKERFILE="TransAxx.Dockerfile"
VERSION="1.0"

# Build Image Tag
IMAGE_TAG="dimdano/transaxx:${VERSION}"

echo "Building TransAxx image..."

# Change to the directory containing the Dockerfile
cd "$(dirname "$0")"/../

docker build -f docker/$DOCKERFILE --tag=$IMAGE_TAG .
