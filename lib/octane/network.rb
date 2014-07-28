module Octane

	class LayerDelegator
		include Enumerable

		def initialize(net, layer)
			@layer=net.layers[layer]
			@net=net
			@layer_number=layer
		end

		def each
			@layer.each {|u| yield u}
		end


		def unit(unit_number)
			return @layer[unit_number]
		end



		def disable(units)
			@net.disable_unit(@layer_number, units)
		end

		def enable(units)
			@net.enable_unit(@layer_number,units)
		end

		def enable_all
			@net.enable_unit(@layer_number,(0...@layer.size).to_a)
		end

		def size
			@layer.size
		end

	end

	class Network
		attr_accessor :layers
		attr_accessor :verbose
		
		def initialize(learning_rate=1, weight_decay=0.0)
			@layers=[]
			@learning_rate=learning_rate
			@weight_decay=weight_decay
		end
	 
		def add_layer(num_neurons, type=:tanh)
			previous_layer=@layers.last
			cls=@layers.last.nil? ? InputNeuron : Neuron
			@layers<<num_neurons.times.each.map{|neuron_id| cls.new(type, "Neuron #{@layers.size}-#{neuron_id}")}
			
			unless previous_layer.nil?
				arity=previous_layer.length
				layer=@layers.last

				lims=1/Math.sqrt(previous_layer.length)

				layer.each {|neuron|
					neuron.input_weights=Array.new(arity) {lims*(2.0*rand-1.0) }
					neuron.input_weights << 0.0

					neuron.weight_changes=Array.new(arity,0.0)
					neuron.weight_changes << 0.0
				}
			end
		end

	 
		def forward_pass(input)
			previous_layer=nil
			@layers.each_with_index{ |layer, layer_number|
				if layer_number==0
					previous_layer=@layers[0].each_with_index.map{|input_neuron, input_neuron_number| input_neuron.set_squashed(input[input_neuron_number]);  input_neuron.disabled ? 0.0 : input[input_neuron_number] }
				else			
					cl=previous_layer.clone+[0.0] # add bias
					layer_outputs=layer.each_with_index.map{|neuron, index| 
														outp=neuron.output(cl)
														puts "Calculating output of neuron #{index} on layer #{layer_number}. Inputs: #{cl}. Output: #{outp}" if @verbose; 
														outp
															}
					previous_layer=layer_outputs
				end

			}
			previous_layer
		end

	 
		def test_classification(data, verbose=false)
			correctly_classified=0.0
			data.each{|example,output|
				# puts "#{example}. Expected: #{output}. Got: #{forward_pass(example)}" if @verbose
				result=forward_pass(example)
				target_class=output.index(output.max)
				result_class=result.index(result.max)
				correctly_classified+=1 if target_class==result_class
			}
	 
			puts "Precision: #{correctly_classified/data.length}" if @verbose
		end

		def test_regression(data, verbose=false)
			err=0
			data.each{|example,output|
				result=forward_pass(example)
				err_datum=output.zip(result).map{|p,o| (p-o)**2}.inject(0){|res, val| res+=val}
				puts "Input: #{example}. Expected: #{output}. Result: #{result}. Error: #{err_datum}" if verbose if @verbose
				err+=err_datum
			}
			
			puts "Error: #{ err/(2*data.length)}" if verbose if @verbose

			return err/(2*data.length)
		end

		def get_activations(data, squashed=false, layer=nil)
			layer=@layers.size-1 unless layer
			
			data.map{|example,output| forward_pass(example); @layers[layer].map{|neuron| squashed ? neuron.last_squashed : neuron.last_activity}}
		end

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

		# def update_weights
		# 	(0..@layers.size-1).to_a.reverse.each_cons(2){ |layer, previous_layer|
		# 		puts "<<<Updating weights between #{previous_layer} and #{layer}>>>" if @verbose
		# 		@layers[layer].each_with_index{ |neuron,index|
		# 			unless neuron.disabled
		# 				weight_changes=@layers[previous_layer].each_index.map{|d| neuron.input_disabled?(d) ? 0.0 : (@layers[previous_layer][d].last_squashed * neuron.delta * @learning_rate)}
		# 				bias_change = @learning_rate*neuron.delta
		# 				puts "Neuron #{index} on layer #{layer}. Incoming weight changes: #{weight_changes}" if @verbose
		# 				neuron.input_weights=neuron.input_weights.zip(weight_changes+[bias_change]).map{|a,b| a - b-@weight_decay*a}
		# 			end
		# 		}
		# 	}
		# end

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
						#neuron.input_weights=neuron.input_weights.zip(weight_changes+[bias_change]).map{|a,b| a - b-@weight_decay*a}
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

		def enable_all_units
			(0...@layers.size).each {|layer|
				enable_unit(layer, (0...@layers[layer].size).to_a)
			}
		end

		def disable_unit(layer, units)
			if units.is_a?(Array)
				units.map{|unit| disable_unit(layer, unit)}
			else
				unit=units
				unless @layers[layer][unit].disabled
					next_layer=layer+1
					prev_layer=layer-1

					@layers[layer][unit].disabled=true

					@layers[next_layer].each_with_index {|neuron, index|
						neuron.ensure_disabled(unit)
					}
				end
			end

		end

		def enable_unit(layer,units)
			if units.is_a?(Array)
				units.map{|unit| enable_unit(layer, unit)}
			else
				unit=units
				if @layers[layer][unit].disabled
					next_layer=layer+1
					prev_layer=layer-1

					@layers[layer][unit].disabled=false

					@layers[next_layer].each_with_index {|neuron, index|
						neuron.ensure_enabled(unit)
					}
				end
			end
		end

		def train_defaults(options)
			options[:epochs]=options[:epochs] || 10

			options
		end

		def train(dataset, options={})
			options=train_defaults(options)
			dataset=dataset.data if dataset.is_a?(Dataset)

			options[:test_set]=options[:test_set].data if options[:test_set] && options[:test_set].is_a?(Dataset)
			options[:reps]=options[:reps] || dataset.size
			options[:test_period]||=options[:reps]/100

			train_errors=[]	
			test_errors=[]
			before_training_results=[]

			options[:reps].times do |timestep|
				
				if timestep%options[:test_period] ==0 and options[:test_set]
					test_errors << test(options[:test_set])
				end
				# options[:epochs].times do |epoch|
				# 	train_errors<<0.0
				# 	example,expected_output=dataset.sample
				example,expected_output=dataset.sample
				options[:before_training][self, timestep, example, expected_output] if options[:before_training]

				# 	network_output=forward_pass(example)
				# 	example_error=calculate_errors(network_output, expected_output)
				# 	update_weights

				# 	train_errors[epoch]+=example_error
				# 	test_errors[epoch]=test(options[:test_set]) if options[:test_set]
				# end
				train_one([example,expected_output], options)
			end

			# result={train_error: train_errors}
			result={}
			result[:test_error]=test_errors if options[:test_set]
			# result[:before_training]=before_training_results if options[:before_training]

			result
		end

		def train_one(sample, options={})
			example, expected_output=sample
			options[:epochs]||=1
			example_error=0
			options[:epochs].times do |epoch|
				network_output=forward_pass(example)
				example_error=calculate_errors(network_output, expected_output)
				calculate_weight_changes
				apply_weight_changes 1
			end
			example_error
		end

		def train_batch(batch, options={})
			batch.each do |example, expected_output|
				network_output=forward_pass(example)
				example_error=calculate_errors(network_output, expected_output)
				calculate_weight_changes
			end

			apply_weight_changes batch.size
		end

		def eval(example)
			forward_pass(example)		
		end

		def output_layer
			LayerDelegator.new(self, @layers.size-1)
		end
	 
		def input_layer
			LayerDelegator.new(self, 0)
		end

		def hidden_layer(num=0)
			raise "Layer #{num} is not a hidden_layer" if num>=@layers.length || num<0
			LayerDelegator.new(self,1+num)
		end

		def output_layer
			@layers.last
		end

		def input_layer
			@layers.first
		end

		def test(dataset)
			dataset=dataset.data if dataset.is_a?(Dataset)
			test_error=0.0
			dataset.each{|example,output|
				result=forward_pass(example)
				test_error+=result.zip(output).map{|r, o| (r-o)**2}.reduce(:+)
			}
			test_error/(2*dataset.length)
		end

		def input_weights(input)
			@layers.first.map{|neuron| neuron.input_weights[input+1]}
		end

		def hidden_weights(layer, from=true)
			@layers[layer].map{|neuron| neuron.input_weights} if from==true
		end
	 
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

		def copy
			Marshal.load(Marshal.dump(self))
		end
	end
end