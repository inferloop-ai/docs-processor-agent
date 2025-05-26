# Document Processor Agent - CI/CD Pipeline
# Handles testing, building, and deployment

name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    # Run tests daily at 2 AM UTC
    - cron: '0 2 * * *'

env:
  PYTHON_VERSION: '3.11'
  NODE_VERSION: '18'

jobs:
  # Code Quality and Linting
  lint:
    name: Code Quality
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
          
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install black flake8 mypy pylint bandit safety
        pip install -r requirements.txt
        
    - name: Run Black formatter check
      run: black --check --diff .
      
    - name: Run Flake8 linting
      run: flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
      
    - name: Run MyPy type checking
      run: mypy agents/ langgraph_flows/ --ignore-missing-imports
      
    - name: Run Pylint
      run: pylint agents/ langgraph_flows/ --disable=missing-docstring,too-few-public-methods
      
    - name: Run Bandit security check
      run: bandit -r agents/ langgraph_flows/ -f json -o bandit-report.json
      
    - name: Run Safety check
      run: safety check --json --output safety-report.json
      continue-on-error: true
      
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json

  # Unit and Integration Tests
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    needs: lint
    
    strategy:
      matrix:
        python-version: ['3.11', '3.12']
        database: ['sqlite', 'postgresql']
        
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_document_processor
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
          
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-${{ matrix.python-version }}-pip-${{ hashFiles('**/requirements.txt') }}
        
    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y \
          libmagic1 \
          poppler-utils \
          tesseract-ocr \
          build-essential
          
    - name: Install Python dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio pytest-mock
        
    - name: Create test directories
      run: |
        mkdir -p logs uploads temp_uploads data
        
    - name: Set up test environment
      run: |
        cp .env.example .env
        echo "TESTING=true" >> .env
        echo "DATABASE_URL=sqlite:///./test.db" >> .env
        
    - name: Set up PostgreSQL test environment
      if: matrix.database == 'postgresql'
      run: |
        echo "DATABASE_URL=postgresql://postgres:postgres@localhost:5432/test_document_processor" >> .env
        
    - name: Run unit tests
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      run: |
        pytest tests/test_agents.py -v --cov=agents --cov-report=xml --cov-report=html
        
    - name: Run integration tests
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      run: |
        pytest tests/test_integration.py -v --cov-append --cov=. --cov-report=xml --cov-report=html
        
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        
    - name: Archive test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.python-version }}-${{ matrix.database }}
        path: |
          htmlcov/
          .coverage
          pytest-report.xml

  # Docker Build and Test
  docker:
    name: Docker Build
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Build Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: false
        tags: |
          document-processor:latest
          document-processor:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        
    - name: Test Docker image
      run: |
        docker run --rm -d --name test-container \
          -p 8000:8000 \
          -e TESTING=true \
          document-processor:latest
        
        # Wait for container to start
        sleep 10
        
        # Test health endpoint
        curl -f http://localhost:8000/health || exit 1
        
        # Stop container
        docker stop test-container
        
    - name: Docker security scan
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: document-processor:latest
        format: 'sarif'
        output: 'trivy-results.sarif'
        
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  # Performance Tests
  performance:
    name: Performance Tests
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install locust pytest-benchmark
        
    - name: Run performance tests
      run: |
        python -m pytest tests/test_performance.py --benchmark-only --benchmark-json=benchmark.json
        
    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: benchmark.json

  # Security Scanning
  security:
    name: Security Scan
    runs-on: ubuntu-latest
    needs: lint
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Run CodeQL Analysis
      uses: github/codeql-action/init@v2
      with:
        languages: python
        
    - name: Autobuild
      uses: github/codeql-action/autobuild@v2
      
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2

  # Deploy to Development
  deploy-dev:
    name: Deploy to Development
    runs-on: ubuntu-latest
    needs: [test, docker]
    if: github.ref == 'refs/heads/develop'
    environment: development
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2
        
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v2
      
    - name: Build and push Docker image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: document-processor
        IMAGE_TAG: dev-${{ github.sha }}
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:dev-latest
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:dev-latest
        
    - name: Deploy to ECS
      run: |
        aws ecs update-service \
          --cluster document-processor-dev-cluster \
          --service document-processor-dev-service \
          --force-new-deployment

  # Deploy to Production
  deploy-prod:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [test, docker, performance, security]
    if: startsWith(github.ref, 'refs/tags/v')
    environment: production
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID_PROD }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY_PROD }}
        aws-region: us-west-2
        
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v2
      
    - name: Get version from tag
      id: get_version
      run: echo "VERSION=${GITHUB_REF#refs/tags/}" >> $GITHUB_OUTPUT
      
    - name: Build and push Docker image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: document-processor  
        IMAGE_TAG: ${{ steps.get_version.outputs.VERSION }}
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
        
    - name: Deploy to ECS
      env:
        VERSION: ${{ steps.get_version.outputs.VERSION }}
      run: |
        aws ecs update-service \
          --cluster document-processor-prod-cluster \
          --service document-processor-prod-service \
          --force-new-deployment
          
    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        draft: false
        prerelease: false

  # Notify on completion
  notify:
    name: Notify
    runs-on: ubuntu-latest
    needs: [deploy-dev, deploy-prod]
    if: always()
    
    steps:
    - name: Notify Slack on success
      if: success()
      uses: 8398a7/action-slack@v3
      with:
        status: success
        text: '🎉 Document Processor deployment successful!'
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
        
    - name: Notify Slack on failure  
      if: failure()
      uses: 8398a7/action-slack@v3
      with:
        status: failure
        text: '❌ Document Processor deployment failed!'
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}