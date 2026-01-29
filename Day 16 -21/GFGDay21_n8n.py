"""
Day 21: n8n Workflow Automation
Demonstrates workflow automation concepts, pipeline orchestration, and process visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
import os
os.makedirs('outputs', exist_ok=True)

# ========================================
# Step 1: Define Workflow Components
# ========================================
print("Creating workflow automation system...")

class WorkflowNode:
    def __init__(self, name, node_type, processing_time=1):
        self.name = name
        self.node_type = node_type
        self.processing_time = processing_time
        self.inputs = []
        self.outputs = []
        self.execution_history = []
        self.success_count = 0
        self.failure_count = 0
    
    def execute(self, data):
        """Execute the node"""
        try:
            # Simulate processing
            result = {
                'timestamp': datetime.now(),
                'data': data,
                'status': 'success' if np.random.random() > 0.05 else 'failure',
                'processing_time': np.random.uniform(self.processing_time * 0.5, self.processing_time * 1.5)
            }
            
            if result['status'] == 'success':
                self.success_count += 1
            else:
                self.failure_count += 1
            
            self.execution_history.append(result)
            return result
        except Exception as e:
            self.failure_count += 1
            return {'status': 'error', 'error': str(e)}
    
    def get_success_rate(self):
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0

# ========================================
# Step 2: Create Workflow Pipeline
# ========================================
print("Building workflow pipeline...")

# Create nodes
trigger = WorkflowNode('Trigger', 'trigger', 0)
data_fetch = WorkflowNode('Data Fetch', 'http', 2)
transform = WorkflowNode('Transform', 'transform', 1.5)
validate = WorkflowNode('Validate', 'filter', 1)
enrich = WorkflowNode('Enrich', 'transform', 2.5)
store = WorkflowNode('Store', 'database', 1.5)
notify = WorkflowNode('Notify', 'webhook', 0.5)
archive = WorkflowNode('Archive', 'storage', 2)

workflow_nodes = [trigger, data_fetch, transform, validate, enrich, store, notify, archive]

# Define connections
connections = [
    (trigger, data_fetch),
    (data_fetch, transform),
    (transform, validate),
    (validate, enrich),
    (enrich, store),
    (store, [notify, archive]),
]

print(f"Created workflow with {len(workflow_nodes)} nodes")

# ========================================
# Step 3: Simulate Workflow Execution
# ========================================
print("Executing workflow simulations...")

num_executions = 100

for execution in range(num_executions):
    data = {'id': execution, 'value': np.random.random() * 100}
    
    for node in workflow_nodes:
        result = node.execute(data)
        if result['status'] == 'failure':
            break
        data = result.get('data', data)

# ========================================
# Step 4: Workflow Performance Visualization
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Success rates by node
success_rates = [node.get_success_rate() for node in workflow_nodes]
node_names = [node.name for node in workflow_nodes]
colors_nodes = plt.cm.Set3(np.linspace(0, 1, len(workflow_nodes)))

axes[0, 0].barh(node_names, success_rates, color=colors_nodes, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Success Rate', fontweight='bold')
axes[0, 0].set_title('Node Success Rates', fontweight='bold')
axes[0, 0].set_xlim([0, 1])
axes[0, 0].grid(True, alpha=0.3, axis='x')

# Add percentages
for i, (name, rate) in enumerate(zip(node_names, success_rates)):
    axes[0, 0].text(rate + 0.02, i, f'{rate*100:.1f}%', va='center', fontweight='bold')

# 2. Execution count and failures
execution_counts = [node.success_count + node.failure_count for node in workflow_nodes]
failure_counts = [node.failure_count for node in workflow_nodes]

x_pos = np.arange(len(node_names))
width = 0.35

axes[0, 1].bar(x_pos - width/2, execution_counts, width, label='Total Executions', 
              color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 1].bar(x_pos + width/2, failure_counts, width, label='Failures', 
              color='red', alpha=0.7, edgecolor='black')
axes[0, 1].set_ylabel('Count', fontweight='bold')
axes[0, 1].set_title('Node Execution and Failure Counts', fontweight='bold')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(node_names, rotation=45, ha='right')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Processing time distribution
processing_times = {node.name: [] for node in workflow_nodes}
for node in workflow_nodes:
    for record in node.execution_history:
        processing_times[node.name].append(record['processing_time'])

data_for_violin = [processing_times[name] for name in node_names]
parts = axes[1, 0].violinplot(data_for_violin, positions=range(len(node_names)), showmeans=True)
axes[1, 0].set_xticks(range(len(node_names)))
axes[1, 0].set_xticklabels(node_names, rotation=45, ha='right')
axes[1, 0].set_ylabel('Processing Time (seconds)', fontweight='bold')
axes[1, 0].set_title('Node Processing Time Distribution', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Workflow statistics
stats_text = f"""
WORKFLOW AUTOMATION SUMMARY
{'='*50}

PIPELINE STRUCTURE:
  Nodes: {len(workflow_nodes)}
  Connections: {len(connections)}
  Parallel Stages: 1

EXECUTION STATISTICS:
  Total Simulations: {num_executions}
  
NODE PERFORMANCE:
"""

for node in workflow_nodes:
    total_exec = node.success_count + node.failure_count
    success_rate = node.get_success_rate() * 100
    avg_time = np.mean([r['processing_time'] for r in node.execution_history]) if node.execution_history else 0
    stats_text += f"\n  {node.name}:\n"
    stats_text += f"    Success Rate: {success_rate:.1f}%\n"
    stats_text += f"    Avg Time: {avg_time:.2f}s\n"

overall_success = sum([n.success_count for n in workflow_nodes]) / (sum([n.success_count + n.failure_count for n in workflow_nodes])) * 100
stats_text += f"\n\nOVERALL SUCCESS RATE: {overall_success:.1f}%"

axes[1, 1].text(0.05, 0.95, stats_text, fontsize=10, family='monospace',
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('outputs/workflow_performance.png', dpi=300, bbox_inches='tight')
plt.close()

print("Workflow performance plot saved!")

# ========================================
# Step 5: Workflow Execution Timeline
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Execution timeline (sample from first node)
timestamps = []
processing_times_list = []
statuses = []

for record in data_fetch.execution_history[:50]:
    if 'processing_time' in record:
        timestamps.append(pd.Timestamp(record['timestamp']))
        processing_times_list.append(record['processing_time'])
        statuses.append(1 if record['status'] == 'success' else 0)

colors_status = ['green' if s == 1 else 'red' for s in statuses]
axes[0, 0].scatter(range(len(processing_times_list)), processing_times_list, 
                  c=colors_status, s=100, alpha=0.6, edgecolors='black')
axes[0, 0].set_xlabel('Execution Sequence', fontweight='bold')
axes[0, 0].set_ylabel('Processing Time (seconds)', fontweight='bold')
axes[0, 0].set_title(f'{data_fetch.name} - Execution Timeline', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 2. Success rate over time (rolling window)
window_size = 10
rolling_success = []
for node in workflow_nodes:
    recent_successes = [1 if r['status'] == 'success' else 0 for r in node.execution_history[-window_size:]]
    success_rate = np.mean(recent_successes) if recent_successes else 0
    rolling_success.append(success_rate)

axes[0, 1].bar(node_names, rolling_success, color=colors_nodes, edgecolor='black', alpha=0.7)
axes[0, 1].set_ylabel('Recent Success Rate', fontweight='bold')
axes[0, 1].set_title(f'Recent Success Rate (Last {window_size} Executions)', fontweight='bold')
axes[0, 1].set_ylim([0, 1])
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Bottleneck analysis
avg_times = [np.mean([r['processing_time'] for r in node.execution_history]) if node.execution_history else 0 for node in workflow_nodes]
axes[1, 0].barh(node_names, avg_times, color='coral', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Average Processing Time (seconds)', fontweight='bold')
axes[1, 0].set_title('Bottleneck Analysis - Node Processing Times', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Highlight bottleneck
max_idx = np.argmax(avg_times)
axes[1, 0].patches[max_idx].set_facecolor('red')

# 4. Workflow throughput
throughput_by_node = []
for node in workflow_nodes:
    total_time = sum([r['processing_time'] for r in node.execution_history])
    throughput = len(node.execution_history) / total_time if total_time > 0 else 0
    throughput_by_node.append(throughput)

axes[1, 1].plot(node_names, throughput_by_node, marker='o', linewidth=2, markersize=8, color='purple')
axes[1, 1].fill_between(range(len(node_names)), throughput_by_node, alpha=0.3, color='purple')
axes[1, 1].set_ylabel('Throughput (executions/second)', fontweight='bold')
axes[1, 1].set_title('Node Throughput Analysis', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('outputs/workflow_execution_timeline.png', dpi=300, bbox_inches='tight')
plt.close()

print("Workflow execution timeline plot saved!")

# ========================================
# Step 6: Error and Failure Analysis
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Failure rate heatmap over time
failure_rates_timeline = []
nodes_sample = workflow_nodes[::2]  # Sample every 2nd node

for node in nodes_sample:
    rates = []
    window = 20
    for i in range(window, len(node.execution_history)):
        recent = node.execution_history[i-window:i]
        failure_rate = sum(1 for r in recent if r['status'] == 'failure') / window
        rates.append(failure_rate)
    failure_rates_timeline.append(rates)

failure_matrix = np.array(failure_rates_timeline)
im = axes[0, 0].imshow(failure_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.2)
axes[0, 0].set_yticks(range(len(nodes_sample)))
axes[0, 0].set_yticklabels([n.name for n in nodes_sample])
axes[0, 0].set_xlabel('Execution Window', fontweight='bold')
axes[0, 0].set_title('Failure Rate Over Time', fontweight='bold')
plt.colorbar(im, ax=axes[0, 0], label='Failure Rate')

# 2. Error types distribution
error_types = defaultdict(int)
for node in workflow_nodes:
    for record in node.execution_history:
        if record['status'] == 'failure':
            error_types[f'{node.name}_Error'] += 1

if error_types:
    error_names = list(error_types.keys())
    error_counts = list(error_types.values())
    axes[0, 1].bar(range(len(error_names)), error_counts, color='red', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xticks(range(len(error_names)))
    axes[0, 1].set_xticklabels(error_names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Error Count', fontweight='bold')
    axes[0, 1].set_title('Error Distribution by Node', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Node reliability scores
reliability_scores = []
for node in workflow_nodes:
    # Reliability = success rate * (1 - normalized processing time variance)
    success_rate = node.get_success_rate()
    times = [r['processing_time'] for r in node.execution_history]
    if times:
        time_variance = np.std(times) / np.mean(times) if np.mean(times) > 0 else 0
        reliability = success_rate * (1 - time_variance / 10)
    else:
        reliability = success_rate
    reliability_scores.append(max(0, reliability))

axes[1, 0].bar(node_names, reliability_scores, color=colors_nodes, edgecolor='black', alpha=0.7)
axes[1, 0].set_ylabel('Reliability Score', fontweight='bold')
axes[1, 0].set_title('Node Reliability Scores', fontweight='bold')
axes[1, 0].set_ylim([0, 1])
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Workflow health summary
from collections import defaultdict
health_text = f"""
WORKFLOW HEALTH REPORT
{'='*50}

OVERALL METRICS:
  Total Executions: {sum([n.success_count + n.failure_count for n in workflow_nodes])}
  Successful: {sum([n.success_count for n in workflow_nodes])}
  Failed: {sum([n.failure_count for n in workflow_nodes])}
  Success Rate: {overall_success:.1f}%

CRITICAL NODES (Low Reliability):
"""

sorted_nodes = sorted(zip(workflow_nodes, reliability_scores), key=lambda x: x[1])
for node, score in sorted_nodes[:3]:
    health_text += f"\n  {node.name}: {score:.2f}\n"

health_text += f"\n\nRECOMMENDATIONS:\n"
health_text += f"  1. Review error handling in low-reliability nodes\n"
health_text += f"  2. Optimize slow nodes for throughput\n"
health_text += f"  3. Add retry logic for critical paths\n"
health_text += f"  4. Monitor {workflow_nodes[np.argmax(avg_times)].name} (bottleneck)\n"

axes[1, 1].text(0.05, 0.95, health_text, fontsize=10, family='monospace',
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('outputs/workflow_health_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Workflow health analysis plot saved!")

# ========================================
# Step 7: Workflow Optimization Recommendations
# ========================================
fig, ax = plt.subplots(figsize=(14, 10))

optimization_report = f"""
{'='*70}
n8n WORKFLOW AUTOMATION - FINAL OPTIMIZATION REPORT
{'='*70}

WORKFLOW CONFIGURATION:
  • Total Nodes: {len(workflow_nodes)}
  • Pipeline Stages: {len(connections)}
  • Parallel Execution Points: 1 (Store -> Notify/Archive)

PERFORMANCE SUMMARY:
  • Overall Success Rate: {overall_success:.1f}%
  • Average Processing Time: {np.mean(avg_times):.2f}s
  • Bottleneck Node: {workflow_nodes[np.argmax(avg_times)].name} ({max(avg_times):.2f}s)
  • Fastest Node: {workflow_nodes[np.argmin(avg_times)].name} ({min(avg_times):.2f}s)

NODE RELIABILITY RANKING:
"""

sorted_reliability = sorted([(n, r) for n, r in zip(workflow_nodes, reliability_scores)], 
                           key=lambda x: x[1], reverse=True)

for rank, (node, score) in enumerate(sorted_reliability, 1):
    optimization_report += f"  {rank}. {node.name:15s} - {score:.3f}\n"

optimization_report += f"""
OPTIMIZATION RECOMMENDATIONS:

1. IMMEDIATE ACTIONS:
   • Investigate {workflow_nodes[np.argmax(avg_times)].name} processing delays
   • Implement exponential backoff retry logic
   • Add data validation before {enrich.name} node

2. PERFORMANCE TUNING:
   • Consider parallelizing {store.name} and {notify.name} operations
   • Cache frequently accessed data in {transform.name}
   • Optimize database queries in {store.name}

3. RELIABILITY IMPROVEMENTS:
   • Add circuit breaker pattern for external APIs
   • Implement dead-letter queue for failed messages
   • Add monitoring and alerting for nodes below 0.8 reliability

4. SCALING CONSIDERATIONS:
   • Current throughput: ~{np.mean(throughput_by_node):.2f} executions/sec
   • For 10x scale: Consider message queuing (RabbitMQ, Apache Kafka)
   • Implement horizontal scaling for stateless nodes

ESTIMATED IMPROVEMENTS:
  • With optimizations: {overall_success*1.1:.1f}% success rate (+10%)
  • Processing time: -{max(avg_times)-min(avg_times):.2f}s per workflow
  • Throughput increase: +25% with parallelization
"""

ax.text(0.05, 0.95, optimization_report, fontsize=9.5, family='monospace',
       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))
ax.axis('off')

plt.tight_layout()
plt.savefig('outputs/workflow_optimization_report.png', dpi=300, bbox_inches='tight')
plt.close()

print("Workflow optimization report saved!")

print("\n✅ n8n Workflow Automation Complete!")
print("Generated outputs:")
print("  - outputs/workflow_performance.png")
print("  - outputs/workflow_execution_timeline.png")
print("  - outputs/workflow_health_analysis.png")
print("  - outputs/workflow_optimization_report.png")

# Import at the top of the file to avoid undefined names
from collections import defaultdict
