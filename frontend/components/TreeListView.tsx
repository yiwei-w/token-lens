'use client'

import React, { useState } from 'react'
import { ChevronRight, ChevronDown, Copy, ExpandIcon, Minimize2 } from 'lucide-react'

interface TreeNode {
  id: number
  parent_id: number | null
  text: string
  prob: number
  log_prob: number
  entropy: number
  depth: number
  cumulative_text: string
  full_prompt: string
  children: number[]
}

interface TreeData {
  tree_nodes: TreeNode[]
  max_depth_reached: number
  total_nodes: number
}

interface TreeListViewProps {
  data: TreeData | null
  onNodeClick?: (node: TreeNode) => void
  selectedNodeId?: number | null
}

const TreeListView: React.FC<TreeListViewProps> = ({
  data,
  onNodeClick,
  selectedNodeId
}) => {
  const [expandedNodes, setExpandedNodes] = useState<Set<number>>(new Set([0])) // Root expanded by default

  if (!data || !data.tree_nodes || data.tree_nodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        No tree data available
      </div>
    )
  }

  // Create lookup maps
  const nodeMap = new Map<number, TreeNode>()
  data.tree_nodes.forEach(node => nodeMap.set(node.id, node))

  const toggleExpansion = (nodeId: number) => {
    const newExpanded = new Set(expandedNodes)
    if (newExpanded.has(nodeId)) {
      newExpanded.delete(nodeId)
    } else {
      newExpanded.add(nodeId)
    }
    setExpandedNodes(newExpanded)
  }

  const expandAll = () => {
    const allNodeIds = new Set(data.tree_nodes.map(node => node.id))
    setExpandedNodes(allNodeIds)
  }

  const collapseAll = () => {
    setExpandedNodes(new Set([0])) // Keep only root expanded
  }

  const calculatePathProbability = (node: TreeNode): number => {
    let pathProb = 1.0
    let current: TreeNode | undefined = node
    
    while (current && current.parent_id !== null) {
      pathProb *= current.prob
      current = nodeMap.get(current.parent_id)
    }
    
    return pathProb
  }

  const formatToken = (text: string): string => {
    if (text === ' ') return '␣'
    if (text === '\n') return '\\n'
    if (text === '\t') return '\\t'
    return text
  }

  const renderNode = (node: TreeNode, level: number = 0): React.ReactNode => {
    const hasChildren = node.children.length > 0
    const isExpanded = expandedNodes.has(node.id)
    const isSelected = selectedNodeId === node.id
    const pathProb = calculatePathProbability(node)

    // Color coding based on probability
    const getProbabilityColor = (prob: number) => {
      if (prob > 0.7) return 'text-green-600 bg-green-50'
      if (prob > 0.3) return 'text-orange-600 bg-orange-50'
      return 'text-red-600 bg-red-50'
    }

    return (
      <div key={node.id}>
        <div
          className={`flex items-center gap-2 py-1 px-2 hover:bg-gray-100 cursor-pointer rounded ${
            isSelected ? 'bg-blue-100 border border-blue-300' : ''
          }`}
          style={{ paddingLeft: `${level * 20 + 8}px` }}
          onClick={() => onNodeClick?.(node)}
        >
          {/* Expansion toggle */}
          <div className="w-4 h-4 flex items-center justify-center">
            {hasChildren ? (
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  toggleExpansion(node.id)
                }}
                className="hover:bg-gray-200 rounded p-0.5"
              >
                {isExpanded ? (
                  <ChevronDown size={12} />
                ) : (
                  <ChevronRight size={12} />
                )}
              </button>
            ) : null}
          </div>

          {/* Token text */}
          <span 
            className={`font-mono text-sm px-1 py-0.5 rounded ${getProbabilityColor(node.prob)}`}
          >
            {node.depth === 0 ? '🌳' : `"${formatToken(node.text)}"`}
          </span>

          {/* Token probability */}
          <span className="text-xs text-gray-600">
            {node.depth === 0 ? 'Root' : `${(node.prob * 100).toFixed(1)}%`}
          </span>

          {/* Path probability */}
          {node.depth > 0 && (
            <span className="text-xs text-blue-600">
              path: {(pathProb * 100).toFixed(2)}%
            </span>
          )}

          {/* Log probability */}
          {node.depth > 0 && (
            <span className="text-xs text-gray-500">
              log: {node.log_prob.toFixed(3)}
            </span>
          )}

          {/* Copy button */}
          {node.depth > 0 && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                navigator.clipboard.writeText(node.full_prompt)
              }}
              className="ml-auto p-1 hover:bg-gray-200 rounded"
              title="Copy prompt up to this point"
            >
              <Copy size={12} />
            </button>
          )}
        </div>

        {/* Render children */}
        {hasChildren && isExpanded && (
          <div>
            {node.children
              .map(childId => nodeMap.get(childId)!)
              .filter(Boolean)
              .sort((a, b) => b.prob - a.prob) // Sort by probability descending
              .map(child => renderNode(child, level + 1))}
          </div>
        )}
      </div>
    )
  }

  const rootNode = data.tree_nodes.find(node => node.parent_id === null)
  if (!rootNode) {
    return <div className="text-red-500">No root node found</div>
  }

  return (
    <div className="h-full overflow-auto bg-white border rounded">
      <div className="p-3 border-b bg-gray-50">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium">Generation Tree</div>
            <div className="text-xs text-gray-600">
              {data.total_nodes} nodes • {data.max_depth_reached} max depth
            </div>
            <div className="text-xs text-gray-500 mt-1">
              Click nodes to select • Colors show token probability • Sorted by probability
            </div>
          </div>
          <div className="flex gap-1">
            <button
              onClick={expandAll}
              className="p-1.5 hover:bg-gray-200 rounded text-gray-600"
              title="Expand All"
            >
              <ExpandIcon size={14} />
            </button>
            <button
              onClick={collapseAll}
              className="p-1.5 hover:bg-gray-200 rounded text-gray-600"
              title="Collapse All"
            >
              <Minimize2 size={14} />
            </button>
          </div>
        </div>
      </div>
      <div className="p-2">
        {renderNode(rootNode)}
      </div>
    </div>
  )
}

export default TreeListView