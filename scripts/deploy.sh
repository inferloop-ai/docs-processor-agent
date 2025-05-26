#!/bin/bash

# Document Processor Agent - Deployment Script
# Supports multiple deployment targets: local, docker, aws

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
DEPLOYMENT_TARGET="local"
ENVIRONMENT="dev"
BUILD_DOCKER=false
PUSH_TO_ECR=false
UPDATE_INFRASTRUCTURE=false

# Print functions
print_status() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }

# Show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --target TARGET     Deployment target (local|docker|aws) [default: local]"
    echo "  -e, --env ENVIRONMENT   Environment (dev|staging|prod) [default: dev]"
    echo "  -b, --build             Build Docker image"
    echo "  -p, --push              Push Docker image to ECR"
    echo "  -i, --infrastructure    Update AWS infrastructure"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --target local                    # Deploy locally"
    echo "  $0 --target docker --build           # Deploy with Docker"
    echo "  $0 --target aws --env prod --build --push --infrastructure"
    echo ""
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--target)
                DEPLOYMENT_TARGET="$2"
                shift 2
                ;;
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -b|--build)
                BUILD_DOCKER=true
                shift
                ;;
            -p|--push)
                PUSH_TO_ECR=true
                shift
                ;;
            -i|--infrastructure)
                UPDATE_INFRASTRUCTURE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Validate deployment target
    if [[ ! "$DEPLOYMENT_TARGET" =~ ^(local|docker|aws)$ ]]; then
        print_error "Invalid deployment target: $DEPLOYMENT_TARGET"
        exit 1
    fi
    
    # Validate environment
    if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
        print_error "Invalid environment: $ENVIRONMENT"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites for $DEPLOYMENT_TARGET deployment..."
    
    # Common prerequisites
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        print_error ".env file not found. Run setup.sh first."
        exit 1
    fi
    
    if [ ! -f "$PROJECT_DIR/requirements.txt" ]; then
        print_error "requirements.txt not found"
        exit 1
    fi
    
    # Docker prerequisites
    if [[ "$DEPLOYMENT_TARGET" == "docker" ]] || [[ "$BUILD_DOCKER" == true ]]; then
        if ! command -v docker &> /dev/null; then
            print_error "Docker is required but not installed"
            exit 1
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            print_error "Docker Compose is required but not installed"
            exit 1
        fi
    fi
    
    # AWS prerequisites
    if [[ "$DEPLOYMENT_TARGET" == "aws" ]]; then
        if ! command -v aws &> /dev/null; then
            print_error "AWS CLI is required but not installed"
            exit 1
        fi
        
        # Check AWS credentials
        if ! aws sts get-caller-identity &> /dev/null; then
            print_error "AWS credentials not configured"
            exit 1
        fi
        
        print_info "AWS Account: $(aws sts get-caller-identity --query Account --output text)"
        print_info "AWS Region: $(aws configure get region)"
    fi
    
    print_status "Prerequisites check passed"
}

# Build Docker image
build_docker_image() {
    print_info "Building Docker image..."
    
    cd "$PROJECT_DIR"
    
    IMAGE_NAME="document-processor"
    IMAGE_TAG="${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S)"
    FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
    
    docker build -t "$FULL_IMAGE_NAME" .
    docker tag "$FULL_IMAGE_NAME" "${IMAGE_NAME}:latest"
    docker tag "$FULL_IMAGE_NAME" "${IMAGE_NAME}:${ENVIRONMENT}"
    
    print_status "Docker image built: $FULL_IMAGE_NAME"
    
    # Export for other functions
    export DOCKER_IMAGE_NAME="$FULL_IMAGE_NAME"
}

# Push to AWS ECR
push_to_ecr() {
    print_info "Pushing Docker image to AWS ECR..."
    
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    AWS_REGION=$(aws configure get region)
    ECR_REPOSITORY="document-processor"
    ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}"
    
    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" --region "$AWS_REGION" 2>/dev/null || \
    aws ecr create-repository --repository-name "$ECR_REPOSITORY" --region "$AWS_REGION"
    
    # Get ECR login token
    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_URI"
    
    # Tag and push image
    docker tag "${DOCKER_IMAGE_NAME}" "${ECR_URI}:${ENVIRONMENT}"
    docker tag "${DOCKER_IMAGE_NAME}" "${ECR_URI}:latest"
    
    docker push "${ECR_URI}:${ENVIRONMENT}"
    docker push "${ECR_URI}:latest"
    
    print_status "Docker image pushed to ECR: ${ECR_URI}:${ENVIRONMENT}"
    
    # Export for other functions
    export ECR_IMAGE_URI="${ECR_URI}:${ENVIRONMENT}"
}

# Deploy locally
deploy_local() {
    print_info "Deploying locally..."
    
    cd "$PROJECT_DIR"
    
    # Ensure virtual environment exists
    if [ ! -d "venv" ]; then
        print_error "Virtual environment not found. Run setup.sh first."
        exit 1
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install/update dependencies
    pip install -r requirements.txt
    
    # Run database migrations if needed
    if command -v alembic &> /dev/null; then
        alembic upgrade head
    fi
    
    # Start the application
    print_info "Starting Document Processor Agent..."
    print_info "API will be available at: http://localhost:8000"
    print_info "Streamlit UI will be available at: http://localhost:8501"
    print_info "Press Ctrl+C to stop"
    
    # Start in background and get PID
    python app.py --mode server --host 0.0.0.0 --port 8000 &
    API_PID=$!
    
    # Start Streamlit UI
    streamlit run web_ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
    UI_PID=$!
    
    # Function to cleanup on exit
    cleanup() {
        print_info "Stopping services..."
        kill $API_PID $UI_PID 2>/dev/null || true
        exit 0
    }
    
    trap cleanup INT TERM
    
    # Wait for processes
    wait $API_PID $UI_PID
}

# Deploy with Docker
deploy_docker() {
    print_info "Deploying with Docker..."
    
    cd "$PROJECT_DIR"
    
    # Build image if requested
    if [[ "$BUILD_DOCKER" == true ]]; then
        build_docker_image
    fi
    
    # Update docker-compose.yml with environment
    export COMPOSE_PROJECT_NAME="document-processor-${ENVIRONMENT}"
    
    # Start services
    docker-compose down 2>/dev/null || true
    docker-compose up -d
    
    print_status "Docker deployment started"
    print_info "API available at: http://localhost:8000"
    print_info "Streamlit UI available at: http://localhost:8502"
    print_info "Database available at: localhost:5432"
    print_info "ChromaDB available at: http://localhost:8001"
    
    # Show logs
    print_info "Showing logs (Ctrl+C to stop watching):"
    docker-compose logs -f
}

# Update AWS infrastructure
update_aws_infrastructure() {
    print_info "Updating AWS infrastructure..."
    
    cd "$PROJECT_DIR"
    
    STACK_NAME="document-processor-${ENVIRONMENT}"
    TEMPLATE_FILE="aws/cloudformation/infrastructure.yaml"
    
    if [ ! -f "$TEMPLATE_FILE" ]; then
        print_error "CloudFormation template not found: $TEMPLATE_FILE"
        exit 1
    fi
    
    # Check if stack exists
    if aws cloudformation describe-stacks --stack-name "$STACK_NAME" &> /dev/null; then
        print_info "Updating existing stack: $STACK_NAME"
        
        aws cloudformation update-stack \
            --stack-name "$STACK_NAME" \
            --template-body "file://$TEMPLATE_FILE" \
            --parameters \
                ParameterKey=ProjectName,ParameterValue=document-processor \
                ParameterKey=Environment,ParameterValue="$ENVIRONMENT" \
                ParameterKey=DatabasePassword,ParameterValue="$(openssl rand -base64 32)" \
            --capabilities CAPABILITY_IAM
    else
        print_info "Creating new stack: $STACK_NAME"
        
        aws cloudformation create-stack \
            --stack-name "$STACK_NAME" \
            --template-body "file://$TEMPLATE_FILE" \
            --parameters \
                ParameterKey=ProjectName,ParameterValue=document-processor \
                ParameterKey=Environment,ParameterValue="$ENVIRONMENT" \
                ParameterKey=DatabasePassword,ParameterValue="$(openssl rand -base64 32)" \
            --capabilities CAPABILITY_IAM
    fi
    
    print_info "Waiting for stack operation to complete..."
    aws cloudformation wait stack-create-complete --stack-name "$STACK_NAME" 2>/dev/null || \
    aws cloudformation wait stack-update-complete --stack-name "$STACK_NAME"
    
    print_status "AWS infrastructure updated successfully"
    
    # Get stack outputs
    print_info "Stack outputs:"
    aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
        --output table
}

# Deploy to AWS
deploy_aws() {
    print_info "Deploying to AWS..."
    
    # Update infrastructure if requested
    if [[ "$UPDATE_INFRASTRUCTURE" == true ]]; then
        update_aws_infrastructure
    fi
    
    # Build and push Docker image
    if [[ "$BUILD_DOCKER" == true ]]; then
        build_docker_image
    fi
    
    if [[ "$PUSH_TO_ECR" == true ]]; then
        push_to_ecr
    fi
    
    # Update ECS service
    STACK_NAME="document-processor-${ENVIRONMENT}"
    
    # Get ECS cluster and service names from CloudFormation
    ECS_CLUSTER=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query 'Stacks[0].Outputs[?OutputKey==`ECSClusterName`].OutputValue' \
        --output text)
    
    if [ -n "$ECS_CLUSTER" ] && [ "$ECS_CLUSTER" != "None" ]; then
        print_info "Updating ECS service..."
        
        # Force new deployment
        aws ecs update-service \
            --cluster "$ECS_CLUSTER" \
            --service "document-processor-${ENVIRONMENT}-service" \
            --force-new-deployment
        
        print_info "Waiting for deployment to complete..."
        aws ecs wait services-stable \
            --cluster "$ECS_CLUSTER" \
            --services "document-processor-${ENVIRONMENT}-service"
        
        print_status "AWS deployment completed successfully"
        
        # Get load balancer URL
        ALB_DNS=$(aws cloudformation describe-stacks \
            --stack-name "$STACK_NAME" \
            --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDNS`].OutputValue' \
            --output text)
        
        if [ -n "$ALB_DNS" ] && [ "$ALB_DNS" != "None" ]; then
            print_info "Application available at: http://$ALB_DNS"
        fi
    else
        print_warning "ECS cluster not found. Infrastructure may not be deployed yet."
    fi
}

# Health check
health_check() {
    print_info "Running health checks..."
    
    case $DEPLOYMENT_TARGET in
        local)
            curl -f http://localhost:8000/health || print_warning "API health check failed"
            curl -f http://localhost:8501 || print_warning "Streamlit health check failed"
            ;;
        docker)
            curl -f http://localhost:8000/health || print_warning "API health check failed"
            curl -f http://localhost:8502 || print_warning "Streamlit health check failed"
            ;;
        aws)
            if [ -n "$ALB_DNS" ]; then
                curl -f "http://$ALB_DNS/health" || print_warning "AWS health check failed"
            fi
            ;;
    esac
}

# Main deployment function
main() {
    echo -e "${BLUE}🚢 Document Processor Agent Deployment${NC}"
    echo "============================================="
    echo "Target: $DEPLOYMENT_TARGET"
    echo "Environment: $ENVIRONMENT"
    echo "============================================="
    
    parse_args "$@"
    check_prerequisites
    
    case $DEPLOYMENT_TARGET in
        local)
            deploy_local
            ;;
        docker)
            deploy_docker
            ;;
        aws)
            deploy_aws
            ;;
        *)
            print_error "Unsupported deployment target: $DEPLOYMENT_TARGET"
            exit 1
            ;;
    esac
    
    # Run health checks
    sleep 5  # Wait a bit for services to start
    health_check
    
    print_status "Deployment completed successfully!"
}

# Run main function with all arguments
main "$@"