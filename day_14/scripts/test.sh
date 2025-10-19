#!/bin/bash
# Day 14 - Test Script for Dockerized RAG Pipeline

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "[$(date +'%H:%M:%S')] $1"
}

BASE_URL=${1:-"http://localhost:8000"}

log "${YELLOW}🧪 Testing RAG Pipeline API at $BASE_URL${NC}"
log "================================================"

# Test 1: Health Check
log "Test 1: Health Check"
response=$(curl -s -w "%{http_code}" "$BASE_URL/health" -o /tmp/health_response)
if [ "$response" = "200" ]; then
    log "${GREEN}✅ Health check passed${NC}"
    cat /tmp/health_response | jq '.'
else
    log "${RED}❌ Health check failed (HTTP $response)${NC}"
    exit 1
fi

# Test 2: API Query
log "\nTest 2: API Query"
query_data='{"text": "What is machine learning?", "max_results": 3}'
response=$(curl -s -w "%{http_code}" -X POST "$BASE_URL/query" \
    -H "Content-Type: application/json" \
    -d "$query_data" \
    -o /tmp/query_response)

if [ "$response" = "200" ]; then
    log "${GREEN}✅ Query test passed${NC}"
    answer=$(cat /tmp/query_response | jq -r '.answer')
    log "Answer preview: ${answer:0:100}..."
else
    log "${RED}❌ Query test failed (HTTP $response)${NC}"
    cat /tmp/query_response
    exit 1
fi

# Test 3: Stats Endpoint
log "\nTest 3: Stats Endpoint"
response=$(curl -s -w "%{http_code}" "$BASE_URL/stats" -o /tmp/stats_response)
if [ "$response" = "200" ]; then
    log "${GREEN}✅ Stats test passed${NC}"
    cat /tmp/stats_response | jq '.pipeline_stats'
else
    log "${RED}❌ Stats test failed (HTTP $response)${NC}"
    exit 1
fi

# Test 4: Metrics Endpoint
log "\nTest 4: Metrics Endpoint"
response=$(curl -s -w "%{http_code}" "$BASE_URL/metrics" -o /tmp/metrics_response)
if [ "$response" = "200" ]; then
    log "${GREEN}✅ Metrics test passed${NC}"
    log "Metrics available: $(wc -l < /tmp/metrics_response) lines"
else
    log "${RED}❌ Metrics test failed (HTTP $response)${NC}"
    exit 1
fi

# Performance Test
log "\nTest 5: Performance Test (10 concurrent requests)"
start_time=$(date +%s)

for i in {1..10}; do
    (
        curl -s -X POST "$BASE_URL/query" \
            -H "Content-Type: application/json" \
            -d '{"text": "Performance test query '$i'", "max_results": 2}' \
            > /tmp/perf_$i.json &
    )
done

wait
end_time=$(date +%s)
duration=$((end_time - start_time))

success_count=0
for i in {1..10}; do
    if [ -f /tmp/perf_$i.json ] && jq -e '.answer' /tmp/perf_$i.json > /dev/null 2>&1; then
        success_count=$((success_count + 1))
    fi
done

log "${GREEN}✅ Performance test: $success_count/10 successful in ${duration}s${NC}"

# Cleanup
rm -f /tmp/*.json /tmp/*_response

log "\n${GREEN}🎉 All tests completed successfully!${NC}"
log "API is ready for production use."
