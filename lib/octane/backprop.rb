module Octane
module Backpropagation
		def calculate_output_deltas(network_output, expected_output)
			output_deltas=expected_output.zip(network_output, @layers.last).map{ |expected, computed, output_neuron|
				output_neuron.delta=-1*(expected-computed)*output_neuron.derivative(computed) 
			}
			puts "COD: Deltas: #{output_deltas}" if @verbose
		end
	 
		def calculate_errors(network_output, expected_output)
			
			calculate_output_deltas(network_output, expected_output)
	 		propagate_deltas_back

	 		network_output.zip(expected_output).map{|computed, expected| (computed-expected)**2}.reduce(:+)/2
		end

		def propagate_deltas_back
			(0..@layers.size-1).to_a.reverse.each_cons(2){|visited_layer, previous_layer|
				puts "<<<Propagating delta from #{visited_layer} to #{previous_layer}>>>" if @verbose
				@layers[previous_layer].each_with_index{|neuron, index|
					unless neuron.disabled
						outgoing_weights=@layers[visited_layer].map{|neuron| neuron.input_disabled?(index) ? 0.0 : neuron.input_weights[index]}
						puts "PDB: Neuron(#{neuron.type}) #{index} on layer #{previous_layer}. Derivative: #{neuron.derivative(neuron.last_squashed)}. Outgoing weights: #{outgoing_weights}. Deltas on lower level: #{@layers[visited_layer].map(&:delta)}" if @verbose
						neuron.delta=neuron.derivative(neuron.last_squashed)*outgoing_weights.zip(@layers[visited_layer].map(&:delta)).map{|a,b| a*b}.reduce(:+)
						puts "PDB: Setting delta to #{neuron.delta}" if @verbose
					end
				}
			}
		end

		def calculate_weight_changes
			(0..@layers.size-1).to_a.reverse.each_cons(2){ |layer, previous_layer|
				puts "<<<Updating weights between #{previous_layer} and #{layer}>>>" if @verbose
				@layers[layer].each_with_index{ |neuron,index|
					unless neuron.disabled
						weight_changes=@layers[previous_layer].each_index.map{|d| neuron.input_disabled?(d) ? 0.0 : (@layers[previous_layer][d].last_squashed * neuron.delta)}
						bias_change = neuron.delta
						puts "CWC: Neuron #{index} on layer #{layer}. Incoming weight changes: #{weight_changes}" if @verbose
						puts "CWC: Neuron #{index} on layer #{layer}. Current weight changes: #{neuron.weight_changes.inspect}" if @verbose
						neuron.weight_changes=neuron.weight_changes.zip(weight_changes+[bias_change]).map {|a,b| a+b}
						puts "CWC: Neuron #{index} on layer #{layer}. Modified weight changes: #{neuron.weight_changes.inspect}" if @verbose
					end
				}
			}
		end

		def learning_rate
			@learning_rate
		end

		def apply_weight_changes(batch_size=1)
			(1..@layers.size-1).each do |layer|
				@layers[layer].each_with_index do |neuron, index|
					puts "AWC: Neuron #{index} on layer #{layer}. Accumulated weight changes: #{neuron.weight_changes}" if verbose
					neuron.input_weights=neuron.input_weights.zip(neuron.weight_changes).map{|old, derivative| old*(1-@weight_decay) - (learning_rate/batch_size) * derivative}
					neuron.weight_changes=Array.new(neuron.weight_changes.length,0.0)
				end
			end
		end

end
end