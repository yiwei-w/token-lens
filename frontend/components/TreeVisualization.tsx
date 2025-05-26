'use client'

import React, { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'

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

interface TreeVisualizationProps {
  data: TreeData | null
  onNodeClick?: (node: TreeNode) => void
  onNodeHover?: (node: TreeNode | null) => void
}

const TreeVisualization: React.FC<TreeVisualizationProps> = ({
  data,
  onNodeClick,
  onNodeHover
}) => {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [selectedNode, setSelectedNode] = useState<TreeNode | null>(null)

  useEffect(() => {
    if (!data || !data.tree_nodes || data.tree_nodes.length === 0) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove() // Clear previous render

    const container = containerRef.current
    if (!container) return

    const width = container.clientWidth
    const height = container.clientHeight

    // Create hierarchy from flat tree structure
    const nodeMap = new Map<number, TreeNode>()
    data.tree_nodes.forEach(node => nodeMap.set(node.id, node))

    const root = data.tree_nodes[0] // Root node
    const hierarchy = d3.hierarchy(root, (d: TreeNode) => 
      d.children.map(childId => nodeMap.get(childId)!).filter(Boolean)
    )

    // Create tree layout
    const treeLayout = d3.tree<TreeNode>()
      .size([height - 100, width - 200])
      .separation((a, b) => a.parent === b.parent ? 1 : 2)

    const treeData = treeLayout(hierarchy)

    // Create SVG group with zoom behavior
    const g = svg.append('g')

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform)
      })

    svg.call(zoom)

    // Initial transform to center the tree
    const initialTransform = d3.zoomIdentity
      .translate(100, 50)
      .scale(0.8)
    
    svg.call(zoom.transform, initialTransform)

    // Draw links
    const links = g.selectAll('.link')
      .data(treeData.links())
      .enter()
      .append('path')
      .attr('class', 'link')
      .attr('d', d3.linkHorizontal<any, any>()
        .x((d: any) => d.y)
        .y((d: any) => d.x)
      )
      .style('fill', 'none')
      .style('stroke', '#ccc')
      .style('stroke-width', 2)

    // Draw nodes
    const nodes = g.selectAll('.node')
      .data(treeData.descendants())
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', (d: any) => `translate(${d.y},${d.x})`)
      .style('cursor', 'pointer')

    // Add circles for nodes
    nodes.append('circle')
      .attr('r', (d: any) => {
        // Scale radius based on probability
        const baseRadius = 8
        const prob = d.data.prob || 0.1
        return baseRadius + Math.log(prob + 0.01) * 2
      })
      .style('fill', (d: any) => {
        // Color based on probability
        const prob = d.data.prob || 0
        if (prob > 0.7) return '#22c55e' // High prob - green
        if (prob > 0.3) return '#f59e0b' // Medium prob - orange  
        return '#ef4444' // Low prob - red
      })
      .style('stroke', (d: any) => selectedNode?.id === d.data.id ? '#000' : '#fff')
      .style('stroke-width', (d: any) => selectedNode?.id === d.data.id ? 3 : 2)

    // Add text labels
    nodes.append('text')
      .attr('dy', '0.35em')
      .attr('x', (d: any) => d.children ? -15 : 15)
      .style('text-anchor', (d: any) => d.children ? 'end' : 'start')
      .style('font-size', '12px')
      .style('font-family', 'Iosevka, monospace')
      .style('fill', '#333')
      .text((d: any) => {
        // Show token text, handling special characters
        let text = d.data.text || ''
        if (text === ' ') return '␣'
        if (text === '\n') return '\\n'
        if (text === '\t') return '\\t'
        return text.length > 8 ? text.substring(0, 8) + '...' : text
      })

    // Add probability labels
    nodes.append('text')
      .attr('dy', '2em')
      .attr('x', (d: any) => d.children ? -15 : 15)
      .style('text-anchor', (d: any) => d.children ? 'end' : 'start')
      .style('font-size', '10px')
      .style('font-family', 'Iosevka, monospace')
      .style('fill', '#666')
      .text((d: any) => {
        const prob = d.data.prob || 0
        return `${(prob * 100).toFixed(1)}%`
      })

    // Add event listeners
    nodes.on('click', (event: any, d: any) => {
      event.stopPropagation()
      setSelectedNode(d.data)
      onNodeClick?.(d.data)
      
      // Update node styling
      nodes.select('circle')
        .style('stroke', (nodeData: any) => nodeData.data.id === d.data.id ? '#000' : '#fff')
        .style('stroke-width', (nodeData: any) => nodeData.data.id === d.data.id ? 3 : 2)
    })

    nodes.on('mouseenter', (event: any, d: any) => {
      onNodeHover?.(d.data)
      
      // Highlight path from root to this node
      const pathNodes = []
      let current = d
      while (current) {
        pathNodes.push(current.data.id)
        current = current.parent
      }
      
      // Highlight links in path
      links.style('stroke', (linkData: any) => {
        const sourceInPath = pathNodes.includes(linkData.source.data.id)
        const targetInPath = pathNodes.includes(linkData.target.data.id)
        return sourceInPath && targetInPath ? '#ff6b6b' : '#ccc'
      })
      .style('stroke-width', (linkData: any) => {
        const sourceInPath = pathNodes.includes(linkData.source.data.id)
        const targetInPath = pathNodes.includes(linkData.target.data.id)
        return sourceInPath && targetInPath ? 3 : 2
      })
      
      // Highlight nodes in path
      nodes.select('circle')
        .style('stroke', (nodeData: any) => {
          if (pathNodes.includes(nodeData.data.id)) return '#ff6b6b'
          return selectedNode?.id === nodeData.data.id ? '#000' : '#fff'
        })
    })

    nodes.on('mouseleave', () => {
      onNodeHover?.(null)
      
      // Reset highlighting
      links.style('stroke', '#ccc').style('stroke-width', 2)
      nodes.select('circle')
        .style('stroke', (nodeData: any) => selectedNode?.id === nodeData.data.id ? '#000' : '#fff')
        .style('stroke-width', (nodeData: any) => selectedNode?.id === nodeData.data.id ? 3 : 2)
    })

  }, [data, selectedNode, onNodeClick, onNodeHover])

  if (!data || !data.tree_nodes || data.tree_nodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        No tree data available
      </div>
    )
  }

  return (
    <div className="w-full h-full" ref={containerRef}>
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        style={{ background: '#fafafa' }}
      />
      <div className="absolute top-4 left-4 bg-white p-2 rounded shadow text-sm">
        <div>Nodes: {data.total_nodes}</div>
        <div>Max Depth: {data.max_depth_reached}</div>
        <div className="text-xs text-gray-500 mt-1">
          Click nodes to select • Hover to highlight paths
        </div>
      </div>
    </div>
  )
}

export default TreeVisualization