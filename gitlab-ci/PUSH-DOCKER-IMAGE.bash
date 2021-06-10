#!/bin/bash

################################################################################

# Tag the Docker image with the Git short SHA associated to the latest 'Dockerfile' modification.
DOCKER_FILE_HASH=`git log -n 1 --no-merges --pretty=format:%h ../docker/turtlebot3-devel/Dockerfile`
docker tag registry.gitlab.com/emlab/turtlebot3/turtlebot3-devel:latest registry.gitlab.com/emlab/turtlebot3/turtlebot3-devel:${DOCKER_FILE_HASH}

# Authenticate with the Docker image registry on GitLab using a deploy token.
docker login -u "gitlab+deploy-token-481366" -p "qncpjW8Aqrhs2nR6zQdZ" registry.gitlab.com/emlab/turtlebot3

# Push the latest Docker image in the registry.
docker push registry.gitlab.com/emlab/turtlebot3/turtlebot3-devel:${DOCKER_FILE_HASH}
docker push registry.gitlab.com/emlab/turtlebot3/turtlebot3-devel:latest
