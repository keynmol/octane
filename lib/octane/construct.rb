module Octane
module Construct
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
					neuron.input_weights << 0.0 if add_bias

					neuron.disabled_inputs=Array.new(arity,0)
					neuron.disabled_inputs << 0 if add_bias

					neuron.weight_changes=Array.new(arity,0.0)
					neuron.weight_changes << 0.0 if add_bias
				}
			end
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
			LayerDelegator.new(self,@layers.size-1)
		end

		def input_layer
			LayerDelegator.new(self,0)
		end
end
end