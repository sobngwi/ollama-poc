// Build every new commit on main: fresh venv, all tests, coverage report.
// Needs the Coverage plugin; everything else comes with the suggested plugins.
pipeline {
    agent any

    triggers {
        // Jenkins only listens on localhost, so GitHub webhooks cannot reach it:
        // check the repository for new commits every 2 minutes instead.
        pollSCM('H/2 * * * *')
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: '30'))
        disableConcurrentBuilds()
        timeout(time: 30, unit: 'MINUTES')
    }

    environment {
        PIP_DISABLE_PIP_VERSION_CHECK = '1'
    }

    stages {
        stage('Set up Python') {
            steps {
                sh '''
                    python3 -m venv .venv
                    .venv/bin/pip install -q -r requirements-dev.txt
                '''
            }
        }

        stage('Test with coverage') {
            steps {
                sh '''
                    .venv/bin/python -m pytest \
                        --junitxml=reports/junit.xml \
                        --cov --cov-report=term \
                        --cov-report=xml:reports/coverage.xml \
                        --cov-report=html \
                        --cov-fail-under=95
                '''
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'reports/junit.xml'
            recordCoverage(tools: [[parser: 'COBERTURA', pattern: 'reports/coverage.xml']],
                           sourceCodeRetention: 'EVERY_BUILD')
            archiveArtifacts allowEmptyArchive: true, artifacts: 'htmlcov/**'
        }
    }
}
