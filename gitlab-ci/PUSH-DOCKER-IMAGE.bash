#!/bin/bash

################################################################################

# Tag the Docker image with the Git short SHA associated to the latest 'Dockerfile' modification.
DOCKER_FILE_HASH=`git log -n 1 --no-merges --pretty=format:%h ../docker/rulo-devel/Dockerfile`
docker tag registry.gitlab.com/emlab/rulo/rulo-devel:latest registry.gitlab.com/emlab/rulo/rulo-devel:${DOCKER_FILE_HASH}

# Authenticate with the Docker image registry on GitLab using a deploy token.
docker login -u "gitlab+deploy-token-480420" -p "7yeQLs94SfBCQ-c7aZYs" registry.gitlab.com/emlab/rulo

# Push the latest Docker image in the registry.
docker push registry.gitlab.com/emlab/rulo/rulo-devel:${DOCKER_FILE_HASH}
docker push registry.gitlab.com/emlab/rulo/rulo-devel:latest
