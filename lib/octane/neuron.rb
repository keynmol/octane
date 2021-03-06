module Octane
	class Neuron
		attr_accessor :delta
		attr_accessor :input_weights, :weight_changes, :disabled_inputs
		attr_accessor :name
		attr_accessor :last_activity, :last_squashed
		attr_reader :type
		attr_accessor :disabled
 
		# TODO: combine both hashes.
		# automatic derivatives?
		SQUASH_FUNCTIONS={:sigmoid => ->(x){1.0/(1.0+Math.exp(-x))}, 
						   :linear => ->(x){ x },
						   :tanh=> ->(x){Math.tanh(x)}}
		SQUASH_DERIVATIVES={:sigmoid => ->(y){ y*(1-y)}, :linear => ->(y) {1}, :tanh => ->(y){1-y*y}}
 
		def initialize(neuron_type=:tanh, name="unnamed neuron..")
			@type=neuron_type
			@delta=0
			@last_activity=0
			@last_squashed=0
			@disabled_inputs=[]
			@disabled=false
			@name=name
			@weight_changes=[]
		end

		def inputs
			@input_weights[0..-2]
		end

		def to_s
			return "<# #{self.class} #{self.name} Input weights: #{self.input_weights}, Pending weight changes: #{self.weight_changes}> "
		end
 
		def output(values)
			if(values.length != @input_weights.length)
				raise "input values for neuron must be of same dimensionality"
			end
 
			@last_activity=(0...@input_weights.length).map{|input|
				values[input]*@input_weights[input]* (1-@disabled_inputs[input])
			}.reduce(:+)
 
			@last_squashed=squash(@last_activity)
		end

 		
		def squash(val)
			SQUASH_FUNCTIONS[@type][val]
		end

		def derivative(val)
			SQUASH_DERIVATIVES[@type][val]
		end

		def ensure_disabled(input)
			unless input_disabled?(input)
				disable_input(input)
			end
		end

		def disable_input(id)
			@disabled_inputs[id]=1
		end

		def enable_input(id)
			@disabled_inputs[id]=0
		end

		def ensure_enabled(input)
			if input_disabled?(input)
				enable_input(input)
			end
		end

		def input_disabled?(input)
			return @disabled_inputs[input]==1
		end

		def clone
			n=Neuron.new(@type)
			n.input_weights=@input_weights.clone
			n.disabled_inputs=@disabled_inputs.clone
			n
		end

	end

	class InputNeuron < Neuron
		def initialize(neuron_type=:linear, name="unnamed neuron..")
			super(neuron_type, name)
			@type=:linear # we force linearity for input neurons
		end
		def set_squashed(value)
			@last_squashed=value
		end

		def output(values)
			[@last_squashed]
		end

	end

end