
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
		attr_accessor :input_transformation

		include Dropout
		include Backpropagation
		include Test
		include Training
		include Plotting
		include Construct

		def initialize(learning_rate=1, weight_decay=0.0, weight_norm=nil)
			@layers=[]
			@learning_rate=learning_rate
			@weight_decay=weight_decay
			@weight_norm=weight_norm
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

		def get_activations(data, squashed=false, layer=nil)
			layer=@layers.size-1 unless layer
			
			data.map{|example,output| forward_pass(example); @layers[layer].map{|neuron| squashed ? neuron.last_squashed : neuron.last_activity}}
		end

	
		

		def eval(example)
			forward_pass(example)		
		end

	
	 
		def copy
			Marshal.load(Marshal.dump(self))
		end
	end
end