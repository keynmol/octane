module Octane
	class AutoEncoder
		attr_reader :encoder, :decoder
		def initialize(input_size, code_size, options={})
			@encoder=Network.new options
			@encoder.add_layer input_size
			@encoder.add_layer code_size, options[:code_layer_type]

			options.delete(:weight_decay)
			@decoder=Network.new options
			@decoder.add_layer code_size
			@decoder.add_layer input_size, options[:output_layer_type]

			@input_mask=options[:input_mask]
			
		end

		def encode(input)
			@encoder.forward_pass(input)
		end

		def decode(code)
			@decoder.forward_pass(code)
		end

		def reconstruct(input)
			decode(encode(input))
		end

		def train_one(input, output, opts={})
			err=train_decoder(input, output, opts)
			train_encoder(input, output)
			err
		end

		def train_decoder(input, output, opts={})
			input=input.zip(@input_mask).map {|i,m| i*m} if @input_mask
			code=encode(input)
			code=code.zip(opts[:code_mask]).map {|c,m| c*m} if opts[:code_mask]
			@decoder.train_one([code, output])
		end

		def train_encoder(input, output)
			# assumes that train_decoder was called first
			code_errors=@decoder.input_layer.map &:delta
			@encoder.output_layer.zip(code_errors).map {|neuron, delta| neuron.delta=delta}
			@encoder.propagate_deltas_back
			@encoder.calculate_weight_changes
			@encoder.apply_weight_changes 1
		end

		def test(dataset)
			err=0
			error=->(input, reconstruction){input.zip(reconstruction).map {|i, r| (i-r)**2}.reduce(:+)/(2*input.length)}
			dataset.data.each {|example, output|
				example=example.zip(@input_mask).map {|i,m| i*m} if @input_mask
				reconstruction=decode(encode(example))
				err+=error[reconstruction, output]
			}
			err/(2*dataset.size)
		end
	end
end