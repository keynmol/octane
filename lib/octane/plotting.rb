module Octane
module Plotting
def graphviz(labels, type=:snapshot)
			str="digraph graphname {\n"
			labels.each{|k,v|
				str+="node_#{k} [label=#{v}];\n"
			}
			@layers.each_with_index{|layer, layer_number|
				if layer_number!=@layers.length-1
					next_layer=@layers[layer_number+1]
				else
					next_layer=nil
				end

				layer.each_with_index{|neuron, neuron_number|
					neuron_style=neuron.disabled ? "style=filled, color=grey" : "style=solid"
					bias=neuron.class == InputNeuron ? "" : "Bias: #{neuron.input_weights.last.round(5)}<br />"
					desc=neuron.class == InputNeuron ? "Input #{neuron_number}.<br /> Activity: #{neuron.last_squashed.round(5)}" : "<b>#{neuron.name}</b> <br />Type: #{neuron.type}, <br />#{bias} Delta: #{neuron.delta.round(4)} <br />Activity: #{neuron.last_activity.round(4)} <br /> Squashed: #{neuron.last_squashed.round(4)}<br />"
					str+="#{node_prefix(layer_number)}_#{neuron_number} [#{neuron_style}, label=<#{desc}>];"
					if next_layer
						next_layer.each_with_index {|next_neuron, next_neuron_number|
							style=(next_neuron.disabled || next_neuron.input_disabled?(neuron_number)) ? "dashed" : "solid"
							str+="\t#{node_prefix(layer_number)}_#{neuron_number} -> #{node_prefix(layer_number+1)}_#{next_neuron_number} [style=#{style}, label=\"#{next_neuron.input_weights[neuron_number].round(5)}\"];\n"
						}
					end
				}
			}
	 
			str+="}"
		end

		def node_prefix(layer_number)
			if layer_number==0
				node_prefix="input"
			elsif layer_number==@layers.length-1
				node_prefix="output"
			else
				node_prefix="hidden_#{layer_number}"
			end

			node_prefix
		end

end
end