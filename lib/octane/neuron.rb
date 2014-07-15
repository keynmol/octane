module Octane
	class Neuron
		attr_accessor :delta
		attr_accessor :input_weights
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
 
		def initialize(neuron_type=:sigmoid, name="unnamed neuron..")
			@type=neuron_type
			@delta=0
			@last_activity=0
			@last_squashed=0
			@disabled_inputs=[]
			@disabled=false
			@name=name
		end

		def to_s
			return "<# #{self.class} #{self.name} Input weights: #{self.input_weights}> "
		end
 
		def output(values)
			if(values.length != @input_weights.length)
				raise "input values for neuron must be of same dimensionality"
			end
 
			@last_activity=(0...@input_weights.length).map{|input|
				input_disabled?(input) ? 0.0 : values[input]*@input_weights[input]
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
			unless @disabled_inputs.include?(input)
				@disabled_inputs<<input
			end
		end

		def ensure_enabled(input)
			if @disabled_inputs.include?(input)
				@disabled_inputs.delete(input)
			end
		end

		def input_disabled?(input)
			return @disabled_inputs.include?(input)
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